import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from CNNImageClassifier import CNNImageClassifier
from CNNAttentionImageClassifier import CNNAttentionImageClassifier
from CNNAttentionTransformerClassifier import CNNAttentionTransformerClassifier
from data_loader import get_dataloaders
import torch.nn as nn

def train_model(train_params, epoch, model_path):
    model = train_params['model']
    model.train()
    device = train_params['device']
    model.to(device)
    running_loss = []
    running_corrects = 0
    total_samples = 0
    pbar = tqdm(total=len(train_params['train_loader']))

    loss_fn = train_params['loss_fn']
    train_loader = train_params['train_loader']
    val_loader = train_params['val_loader']
    optimizer = train_params['optim']

    for img, label in train_loader:
        img_batch, label_batch = img.to(device), label.to(device)
        out_batch = model(img_batch)
        loss = loss_fn(out_batch, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        
        # Calculate accuracy
        _, preds = torch.max(out_batch, 1)
        running_corrects += torch.sum(preds == label_batch.data)
        total_samples += label_batch.size(0)

        pbar.update(1)
        pbar.set_description(
            f'Epoch {epoch}/{train_params["training_epoch"]}  Loss: {np.mean(running_loss):.4f}  Acc: {running_corrects.double() / total_samples:.4f}'
        )

    # Validation loss and accuracy calculation
    tmp_loss = []
    tmp_corrects = 0
    tmp_total = 0
    model.eval()
    with torch.no_grad():
        for img, label in val_loader:
            img_batch, label_batch = img.to(device), label.to(device)
            out_batch = model(img_batch)
            loss = loss_fn(out_batch, label_batch)
            tmp_loss.append(loss.item())
            
            _, preds = torch.max(out_batch, 1)
            tmp_corrects += torch.sum(preds == label_batch.data)
            tmp_total += label_batch.size(0)

    val_loss = np.mean(tmp_loss)
    val_acc = tmp_corrects.double() / tmp_total
    pbar.set_description(f'Epoch {epoch} Validation Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}')
    pbar.close()
    
    model_states = {
        "epoch": epoch, 
        "state_dict": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
        "loss": running_loss
    }
    torch.save(model_states, model_path)

    return val_loss, model_states

if __name__ == "__main__":
    print('------------Starting Retinal Image Training------------\n')

    parser = argparse.ArgumentParser(description='Training script for Retinal Image Classification')
    parser.add_argument('--Device', type=int, help='CUDA device for training', default=0)
    parser.add_argument('--lr', type=float,  help='Learning rate for the optimizer', default=1e-4)
    parser.add_argument('--BatchSize', help='Size of the minibatch', type=int, default=32)
    parser.add_argument('--DataDir', help='Directory containing the images and labels', type=str, required=True)
    parser.add_argument('--ModelOut', help='Destination for saving the trained model', type=str, required=True)
    parser.add_argument('--Epoch', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--SplitDir', help='Directory to save or load data splits', type=str, default='splits/')
    args = parser.parse_args()

    if args.ModelOut is None:
        print('Error: Please provide model output directory')
        sys.exit(1)
    elif not os.path.isdir(args.ModelOut):
        os.makedirs(args.ModelOut)

    # Dataset loading with train/validation/test splits and saving/loading splits
    train_loader, val_loader, test_loader = get_dataloaders(
        img_dir=args.DataDir, 
        label_path=os.path.join(args.DataDir, 'labels.pt'), 
        split_dir=args.SplitDir, 
        batch_size=args.BatchSize
    )

    # Define model architectures
    model_architectures = {
        'CNNAttention': CNNAttentionImageClassifier,
        'CNNTransformer': CNNAttentionTransformerClassifier,
        'CNN': CNNImageClassifier
    }

    # Hyperparameter values for retinal image classification
    kernels = [5]
    filters = [64]
    layers = [5]
    learning_rates = [1e-4]
    pool_sizes = [2]

    # Device configuration
    device = torch.device(f"cuda:{args.Device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    for model_name, model_class in model_architectures.items():
        print(f"\nTraining model: {model_name}")
        for kernel in kernels:
            for filter_size in filters:
                for layer_count in layers:
                    for pool in pool_sizes:
                        for lr in learning_rates:
                            # Instantiating the model based on the architecture
                            if model_name == 'CNN':
                                # model = model_class(
                                #     input_size=(3, 256, 256), 
                                #     num_classes=4, 
                                #     num_filters=filter_size,
                                #     kernel_size=kernel, 
                                #     pool_size=pool, 
                                #     num_layers=layer_count
                                # )
                                continue
                            elif model_name == 'CNNAttention':
                                model = model_class(
                                    input_size=(3, 256, 256),
                                    num_classes=4,
                                    num_filters=filter_size,
                                    kernel_size=kernel,
                                    pool_size=pool,
                                    num_conv_layers=layer_count,
                                    attention_reduction=16,
                                    attention_kernel_size=3,
                                    fc_size=128,
                                    dropout=0.3
                                )
                                # continue
                            elif model_name == 'CNNTransformer':
                                model = model_class(
                                    input_size=(3, 256, 256),  # Updated input size
                                    num_classes=4, 
                                    num_filters=filter_size,
                                    kernel_size=kernel, 
                                    pool_size=pool, 
                                    num_conv_layers=layer_count, 
                                    transformer_dim=128, 
                                    nhead=4, 
                                    num_transformer_layers=3, 
                                    dim_feedforward=256, 
                                    dropout=0.3, 
                                    fc_size=128
                                )
                                # continue
                            model = model.to(device)
                            
                            # optimizer and loss function
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            loss_fn = nn.CrossEntropyLoss()
                            
                            train_param_dict = {
                                'model': model,
                                'optim': optimizer,
                                'loss_fn': loss_fn,
                                'train_loader': train_loader,
                                'val_loader': val_loader,
                                'device': device,
                                'training_epoch': args.Epoch
                            }

                            # unique output path for each model configuration
                            output_path = os.path.join(
                                args.ModelOut, 
                                f'{model_name}_L{layer_count}_F{filter_size}_K{kernel}_P{pool}_lr{lr}'
                            )
                            os.makedirs(output_path, exist_ok=True)
                            output_file = os.path.join(output_path, 'model.pth')

                            patience = 4
                            best_val_loss = np.inf
                            best_epoch = 0
                            best_model = None
                            counter = 0

                            for epoch in range(1, args.Epoch + 1):
                                val_loss, current_model = train_model(
                                    train_params=train_param_dict, 
                                    epoch=epoch, 
                                    model_path=output_file
                                )
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    best_epoch = epoch
                                    counter = 0
                                    best_model = current_model
                                else:
                                    counter += 1
                                    if counter > patience:
                                        print(f'Early stopping at epoch {epoch}')
                                        break

                            print(f'Training for {model_name} stopped after {best_epoch} epochs. Saving best model.')
                            torch.save(best_model, output_file)