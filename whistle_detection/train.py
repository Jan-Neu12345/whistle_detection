import argparse
import os

import torch
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from lion_pytorch import Lion
from dataset import AudioDataset
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from utils import print_environment_info, provide_determinism

from whistle_detection import get_model, worker_seed_set

try:
    import wandb
except ImportError:
    print("Wandb is an optional dependency and currently not installed")

def run():
    print_environment_info()

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to dataset json file")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="Number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of the batches"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Interval of epochs between saving model weights",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory in which the checkpoints are stored",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.8,
        help="Fraction of dataset to use for training",
    )
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=1,
        help="Interval of epochs between evaluations on validation set",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Evaluation: Whistle confidence threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Makes results reproducible. Set -1 to disable.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=10_000,
        help="Targeted sample rate of the audio files",
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=1,
        help="Duration of the chunks in seconds",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable weights and biases (wandb) logging",
    )
    args = parser.parse_args()
    print(args)

    if not args.disable_wandb:
        wandb.init(
            project="bitbots_whistle_detection",
            entity="bitbots",
            config={
                "dataset_path": args.dataset_path,
                "epochs": args.epochs,
                "n_cpu": args.n_cpu,
                "batch_size": args.batch_size,
                "checkpoint_interval": args.checkpoint_interval,
                "checkpoint_dir": args.checkpoint_dir,
                "train_test_split": args.train_test_split,
                "evaluation_interval": args.evaluation_interval,
                "learning_rate": args.learning_rate,
                "conf_threshold": args.conf_threshold,
                "seed": args.seed,
                "sample_rate": args.sample_rate,
                "chunk_duration": args.chunk_duration,
            },
        )

    if args.seed != -1:
        provide_determinism(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AudioDataset(
        args.dataset_path,
        args.sample_rate,
        args.chunk_duration,
        train_mode=True,
        train_test_split=args.train_test_split,
        seed=args.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        worker_init_fn=worker_seed_set,
    )

    validation_dataset = AudioDataset(
        args.dataset_path,
        args.sample_rate,
        args.chunk_duration,
        train_mode=False,
        train_test_split=args.train_test_split,
        seed=args.seed,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        worker_init_fn=worker_seed_set,
    )

    num_true = 0
    total = 0
    for _, (_, label) in enumerate(tqdm.tqdm(train_dataloader)):
        num_true += label[:, 1].int().sum().item()
        total += len(label)

    weight_true = total / (num_true * 2)
    weight_false = total / ((total - num_true) * 2)

    # print(weight_true)
    # print(weight_false)

    # bce = BCEWithLogitsLoss(weight=torch.Tensor([weight_false, weight_true]).to(device))
    bce = BCEWithLogitsLoss()

    model = get_model(device)

    params = [p for p in model.parameters() if p.requires_grad]

    #optimizer = optim.Adam(
    #    params,
    #    lr=args.learning_rate,
    #)
    optimizer = optim.Adam(
        params,
        lr=args.learning_rate,
    )

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs + 1):
        model.train()

        #for batch_i, (spectograms, labels) in enumerate(
        for spectograms, labels in tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch}"
        ):
            spectograms = Variable(spectograms.to(device), requires_grad=False)
            labels = Variable(labels.float().to(device), requires_grad=False)

            outputs = model(spectograms)
            outputs = torch.squeeze(outputs)
            # print(outputs.size())
            print(labels)

            # # get two tensors with ones at each classes instance
            # ones_at_true = label.int()
            # ones_at_false = (torch.ones_like(ones_at_true) - ones_at_true).double()
            # ones_at_true = ones_at_true.double()
            # ones_at_true *= weight_true
            # ones_at_false *= weight_false
            # weights = ones_at_true + ones_at_false
            

            loss = bce(outputs, labels)
            loss.backward()
            #print(loss)

            if not args.disable_wandb:
                wandb.log({"train_loss": loss.item()})

            ###############
            # Run optimizer
            ###############

            #TODO Optimizer überdenken / Kein Optimizer = mehr TP, aber keine Verbesserung nach Epoche 1 wtf?!?
            optimizer.step()
            optimizer.zero_grad()

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)
        
        print(labels)
        print(outputs)
        # train conf matr
        
        results = []
        all_true_positives = 0
        all_false_positives = 0
        all_true_negatives = 0
        all_false_negatives = 0

        all_true_positives += (outputs[labels[:, 0] == 1][:, 0] > outputs[labels[:, 0] == 1][:, 1]).float().sum().item()
        all_false_positives += (outputs[labels[:, 0] == 0][:, 0] > outputs[labels[:, 0] == 0][:, 1]).float().sum().item()
        all_false_negatives += (outputs[labels[:, 0] == 1][:, 0] <= outputs[labels[:, 0] == 1][:, 1]).float().sum().item()
        all_true_negatives += (outputs[labels[:, 0] == 0][:, 0] <= outputs[labels[:, 0] == 0][:, 1]).float().sum().item()
        results.append(float(labels.eq(outputs >= args.conf_threshold).float().mean()))

        all_positives = all_true_positives+all_false_negatives
        all_negatives = all_true_negatives+all_false_positives
        plt.figure()

        #TODO handling für all positives/negatives == 0
        heatmap_normalized = sns.heatmap(
            np.array(
                [
                    [all_true_negatives/(all_negatives) if all_negatives > 0 else 0, all_false_positives/(all_negatives) if all_negatives > 0 else 0],
                    [all_false_negatives/(all_positives) if all_positives > 0 else 0, all_true_positives/(all_positives)  if all_positives > 0 else 0]
                ]
            ),
            annot=True
        )
        plt.savefig(f"conf_train_matr/conf_matrix_epoch{epoch}.png")

        




        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            # Evaluate the model on the validation set
            metrics_output = {
                "mean": evaluate(
                    model, validation_dataloader, args.conf_threshold, device
                )
            }
            plt.savefig(f"conf_matr/conf_matrix_epoch{epoch}.png")

            if not args.disable_wandb:
                wandb.log(metrics_output)
            print(f"---- Evaluation metrics: {metrics_output} ----")


def evaluate(model, dataloader, conf_threshold, device):
    model.eval()
    validate_bce = BCEWithLogitsLoss()
    
    results = []
    all_true_positives = 0
    all_false_positives = 0
    all_true_negatives = 0
    all_false_negatives = 0

    
    for spectograms, labels in tqdm.tqdm(dataloader, desc="Validating"):
        # print(labels.size())
        spectograms = Variable(spectograms.to(device), requires_grad=False)
        labels = Variable(labels.float().to(device), requires_grad=False)

        with torch.no_grad():
            outputs = torch.sigmoid(model(spectograms)).squeeze()
#            validate_loss = validate_bce(outputs, labels)
#        print(validate_loss)

        # print(labels, outputs)
        all_true_positives += (outputs[labels[:, 0] == 1][:, 0] > outputs[labels[:, 0] == 1][:, 1]).float().sum().item()
        all_false_positives += (outputs[labels[:, 0] == 0][:, 0] > outputs[labels[:, 0] == 0][:, 1]).float().sum().item()
        all_false_negatives += (outputs[labels[:, 0] == 1][:, 0] <= outputs[labels[:, 0] == 1][:, 1]).float().sum().item()
        all_true_negatives += (outputs[labels[:, 0] == 0][:, 0] <= outputs[labels[:, 0] == 0][:, 1]).float().sum().item()
        results.append(float(labels.eq(outputs >= conf_threshold).float().mean()))
    
    # heatmap = sns.heatmap(
    #     np.array(
    #         [
    #             [all_true_negatives, all_false_positives],
    #             [all_false_negatives, all_true_positives]
    #         ]
    #     ),
    #     annot=True
    # )
    all_positives = all_true_positives+all_false_negatives
    all_negatives = all_true_negatives+all_false_positives
    plt.figure()
    #TODO handling für all positives/negatives == 0
    heatmap_normalized = sns.heatmap(
        np.array(
            [
                [all_true_negatives/(all_negatives) if all_negatives > 0 else 0, all_false_positives/(all_negatives) if all_negatives > 0 else 0],
                [all_false_negatives/(all_positives) if all_positives > 0 else 0, all_true_positives/(all_positives)  if all_positives > 0 else 0]
            ]
        ),
        annot=True
    )
    # plt.show()

    return results


if __name__ == "__main__":
    run()
