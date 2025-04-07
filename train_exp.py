# train_exp.py
import os
import json
import torch
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
from deep_features import load_json_data, merge_nodes, build_node_features
from dataloader import load_and_build
from graph_construction import compute_edge_weights, create_graph_data
from network import GATv2Model
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_score,
    average_precision_score,
    mean_absolute_error
)

def weighted_emd_loss(logits, target):
    p = torch.softmax(logits, dim=1)
    cdf_p = torch.cumsum(p, dim=1)
    N, C = logits.shape
    device = logits.device
    class_range = torch.arange(C, device=device).unsqueeze(0).repeat(N, 1).float()
    target = target.float().unsqueeze(1)
    cdf_target = (class_range >= target).float()
    loss = torch.mean(torch.abs(cdf_p - cdf_target))
    return loss

def evaluate_model(model_gat, data_val, device, num_classes=6):
    model_gat.eval()
    with torch.no_grad():
        logits = model_gat(data_val)
    val_emd = weighted_emd_loss(logits, data_val.y).item()
    preds = logits.argmax(dim=1).cpu().numpy()
    labels = data_val.y.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(num_classes), zero_division=0
    )
    accuracy = np.mean(preds == labels)
    global_precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    global_precision_weighted = precision_score(labels, preds, average="weighted", zero_division=0)
    from sklearn.preprocessing import label_binarize
    prob = torch.softmax(logits, dim=1).cpu().numpy()
    labels_bin = label_binarize(labels, classes=range(num_classes))
    ap_per_class = []
    for c in range(num_classes):
        ap_c = average_precision_score(labels_bin[:, c], prob[:, c])
        ap_per_class.append(ap_c)
    mae_val = mean_absolute_error(labels, preds)
    metrics_dict = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "ap": ap_per_class,
        "mae": mae_val,
        "accuracy": accuracy,
        "global_precision_macro": global_precision_macro,
        "global_precision_weighted": global_precision_weighted
    }
    return val_emd, metrics_dict

def run_training(seed, config):
    # Set seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"\n===== Starting training with seed {seed} =====")
    
    data_base = "/home/fuad/Work/Projects/hurricane/data"
    fine_tune_transformer = bool(config["fine_tune_transformer"])

    # Load transformer and processor.
    processor = AutoImageProcessor.from_pretrained(config["model_name"])
    model_transformer = AutoModelForImageClassification.from_pretrained(
        config["model_name"], output_hidden_states=True
    )
    if not fine_tune_transformer:
        model_transformer.eval()
        for param in model_transformer.parameters():
            param.requires_grad = False
        print("Transformer is frozen.")
    else:
        model_transformer.train()
        if "MITLL/LADI-v2-classifier-small" in config["model_name"]:
            for name, param in model_transformer.named_parameters():
                if ("bit.embedder" in name or 
                    "bit.encoder.stages.0" in name or 
                    "bit.encoder.stages.1" in name):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print("Small classifier: Transformer partially unfrozen (stages 2,3 and classifier head trainable).")
        elif "MITLL/LADI-v2-classifier-large" in config["model_name"]:
            for name, param in model_transformer.named_parameters():
                if ("swinv2.embeddings" in name or 
                    "swinv2.encoder.layers.0" in name or 
                    "swinv2.encoder.layers.1" in name):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print("Large classifier: Transformer partially unfrozen (layers 2+ and classifier head trainable).")
        else:
            for param in model_transformer.parameters():
                param.requires_grad = True
            print("Transformer unfrozen (default): all layers trainable.")

    # Load training data using the global dataloader.
    train_data = load_and_build(
        data_base, "train",
        processor, model_transformer,
        pooling_mode=config["pooling_mode"],
        fine_tune_transformer=fine_tune_transformer
    )
    nodes_train = train_data["nodes"]
    coords_train = train_data["coords"]
    node_feats_train = train_data["node_feats"]
    gt_damage_train = train_data["gt_damage"]

    edge_index_train, edge_weight_train = compute_edge_weights(
        coords_train, node_feats_train,
        k=int(config["k_neighbors"]),
        alpha=float(config["alpha"]),
        beta=float(config["beta"])
    )
    data_train = create_graph_data(node_feats_train, edge_index_train, edge_weight_train, gt_damage_train)

    # Load validation data.
    val_data = load_and_build(
        data_base, "val",
        processor, model_transformer,
        pooling_mode=config["pooling_mode"],
        fine_tune_transformer=fine_tune_transformer
    )
    nodes_val = val_data["nodes"]
    coords_val = val_data["coords"]
    node_feats_val = val_data["node_feats"]
    gt_damage_val = val_data["gt_damage"]

    edge_index_val, edge_weight_val = compute_edge_weights(
        coords_val, node_feats_val,
        k=int(config["k_neighbors"]),
        alpha=float(config["alpha"]),
        beta=float(config["beta"])
    )
    data_val = create_graph_data(node_feats_val, edge_index_val, edge_weight_val, gt_damage_val)

    in_dim = node_feats_train.shape[1]
    num_classes = 6
    hidden_channels = int(config["hidden_channels"])
    heads = int(config["heads"])
    lr = float(config["lr"])
    model_gat = GATv2Model(
        in_channels=in_dim,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        heads=heads,
        return_attention=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_train = data_train.to(device)
    data_val = data_val.to(device)
    model_gat.to(device)
    model_transformer.to(device)

    params_to_optimize = list(model_gat.parameters())
    if fine_tune_transformer:
        params_to_optimize += list(model_transformer.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-4)

    best_val_emd = float("inf")
    no_improvement = 0
    os.makedirs("./checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"best_model_seed{seed}_{timestamp}.pt"
    best_checkpoint_path = os.path.join("./checkpoints", checkpoint_filename)

    epoch_list = []
    train_emd_list = []
    val_emd_list = []
    global_accuracy_history = []
    global_precision_macro_history = []
    global_precision_weighted_history = []
    global_mae_history = []

    max_epochs = 300
    patience = int(config["patience"])
    for epoch in range(1, max_epochs + 1):
        model_gat.train()
        if fine_tune_transformer:
            model_transformer.train()
        else:
            model_transformer.eval()

        optimizer.zero_grad()
        out_train = model_gat(data_train)
        loss_train = weighted_emd_loss(out_train, data_train.y)
        loss_train.backward()
        optimizer.step()
        train_emd = loss_train.item()

        model_gat.eval()
        model_transformer.eval()
        with torch.no_grad():
            out_val = model_gat(data_val)
            loss_val = weighted_emd_loss(out_val, data_val.y)
        val_emd = loss_val.item()

        val_emd, metrics_dict = evaluate_model(model_gat, data_val, device, num_classes=num_classes)
        epoch_accuracy = metrics_dict["accuracy"]
        epoch_global_precision_macro = metrics_dict["global_precision_macro"]
        epoch_global_precision_weighted = metrics_dict["global_precision_weighted"]
        epoch_mae = metrics_dict["mae"]

        if val_emd < best_val_emd:
            best_val_emd = val_emd
            no_improvement = 0
            checkpoint = {
                "model_gat_state_dict": model_gat.state_dict(),
                "model_transformer_state_dict": model_transformer.state_dict() if fine_tune_transformer else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_emd": val_emd,
                "seed": seed,
                "config": config
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with val_emd = {val_emd:.4f} (seed {seed})")
        else:
            no_improvement += 1

        epoch_list.append(epoch)
        train_emd_list.append(train_emd)
        val_emd_list.append(val_emd)
        global_accuracy_history.append(epoch_accuracy)
        global_precision_macro_history.append(epoch_global_precision_macro)
        global_precision_weighted_history.append(epoch_global_precision_weighted)
        global_mae_history.append(epoch_mae)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{max_epochs}: "
              f"Train EMD = {train_emd:.4f}, Val EMD = {val_emd:.4f}, "
              f"Accuracy = {epoch_accuracy:.4f}, Macro Precision = {epoch_global_precision_macro:.4f}, "
              f"Weighted Precision = {epoch_global_precision_weighted:.4f}, MAE = {epoch_mae:.4f}")

        if no_improvement >= patience:
            print(f"Early stopping after {epoch} epochs (no improvement in {patience} epochs).")
            break

    final_metrics = {
        "seed": seed,
        "best_val_emd": best_val_emd,
        "final_epoch": epoch,
        "global_accuracy": global_accuracy_history[-1],
        "global_precision_macro": global_precision_macro_history[-1],
        "global_precision_weighted": global_precision_weighted_history[-1],
        "global_mae": global_mae_history[-1]
    }
    return final_metrics

def main():
    best_config_path = "./results/best_config.json"
    if not os.path.exists(best_config_path):
        raise FileNotFoundError(f"{best_config_path} not found. Please run hyperparameter tuning first.")
    with open(best_config_path, "r") as f:
        config = json.load(f)

    print("Using best hyperparameters from tune:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    num_runs = 200
    results_list = []
    for seed in range(num_runs):
        run_metrics = run_training(seed, config)
        results_list.append(run_metrics)

    import pandas as pd
    df = pd.DataFrame(results_list)
    os.makedirs("./results", exist_ok=True)
    csv_path = "./results/final_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Final metrics for {num_runs} runs saved to {csv_path}")

if __name__ == "__main__":
    main()
