import os
import torch
import json
import numpy as np
import argparse
import torch.utils.data
import mdtraj as md
from utils import data_normalization, data_denormalization
from model import VAE, VAELoss
from train_eval import train_model, evaluate_model
from weighted_rmsd_fitting import WeightedRMSDFit

def main(args):

    print("####################")
    print("### VAE Analysis ###")
    print("####################\n")

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    hparams_filename = args.hyperparameters.split('/')[-1].split('.')[0]
    condition_folder = os.path.join("Results", "VAE", args.condition)
    save_folder = os.path.join(condition_folder, hparams_filename)
    os.makedirs(save_folder, exist_ok=True)
    
    with open(args.hyperparameters, "r") as f:
        hparams = json.load(f)

    # Save hyperparameters
    with open(os.path.join(save_folder, "job_hyperparameters.json"), "w") as f:
        json.dump(hparams, f)
        json.dump(vars(args), f)
        print("Hyperparameters:\n", hparams, "\n")

    # Weighted RMSD fit
    if args.fit_traj:
        print("Fitting trajectory...")
        fitted_traj = WeightedRMSDFit(args.pdb_path, 
                                    args.xtc_path,  
                                    args.sfactor, 
                                    args.ref_pdb,
                                    hparams["stride"])

        traj = md.Trajectory(fitted_traj, md.load(args.pdb_path).topology)

        # Save fitted trajectory
        traj[0].save(os.path.join(condition_folder,args.condition+'_fit.pdb'))
        traj.save_xtc(os.path.join(condition_folder,args.condition+'_fit.xtc'))
        print("Fitted trajectory saved...\n")
    
    else:
        traj = md.load(args.xtc_path,
                    top=args.pdb_path,
                    stride=hparams["stride"])

    data = traj.xyz

    n_frames, n_atoms, n_xyz = np.shape(data)
    n_features = n_atoms * n_xyz

    data = np.reshape(data, (n_frames, n_features))

    data_norm, max_value, min_value = data_normalization(data)

    np.savez(os.path.join(save_folder, f"{args.condition}_scaler.npz"),
            max_value=max_value,
            min_value=min_value)
    
    print("Normalization scalers saved...\n")

    # Select the data for training and validation steps
    select_train = int(hparams["partition"]*n_frames)
    select_valid = int((1-hparams["partition"])*n_frames)
    train_idx = np.random.choice(len(data_norm), select_train, replace=False)
    valid_idx = np.random.choice(len(data_norm), select_valid, replace=False)

    train = data_norm[train_idx]
    valid = data_norm[valid_idx]

    train_data = torch.FloatTensor(train)
    valid_data = torch.FloatTensor(valid)
    test_data = torch.FloatTensor(data_norm)

    # Transform into a pytorch dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset = train_data,
                                                    batch_size = hparams["batch_size"],
                                                    drop_last=True,
                                                    shuffle = True)
    valid_dataloader = torch.utils.data.DataLoader(dataset = valid_data,
                                                batch_size = hparams["batch_size"],
                                                drop_last=True,
                                                shuffle = False)
    test_dataloader = torch.utils.data.DataLoader(dataset = test_data,
                                                batch_size = hparams["batch_size"],
                                                drop_last=False,
                                                shuffle = False)
    
    # Model Instatiation
    model = VAE(input_dim=n_features,
                nlayers=hparams["layers"],
                latent_dim=hparams["latent_dim"],
                dropout=hparams["dropout"],
                neg_slope=hparams["neg_slope"])

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print("Model loaded...\n")

    model.to(device)
    
    print("MODEL:\n", model, "\n")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = hparams["lr"],
                                weight_decay=hparams["weight_decay"],
                                foreach=False)

    # Loss function
    loss_function = VAELoss(loss_type=hparams["loss_function"]["reconstruction"],
                            reduction=hparams["loss_function"]["reduction"])
    
    if args.train_model:        
        losses = train_model(model,
                        train_dataloader,
                        valid_dataloader,
                        optimizer,
                        loss_function,
                        hparams,
                        save_folder,
                        device)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(save_folder, f"{args.condition}_model_epochs{hparams['epochs']}.pth"))
        torch.save(model.encoder.state_dict(), os.path.join(save_folder, f"en{args.condition}_encoder_epochs{hparams['epochs']}.pth"))
        torch.save(model.decoder.state_dict(), os.path.join(save_folder, f"de{args.condition}_decoder_epochs{hparams['epochs']}.pth"))
        print("Model saved...\n")

    # Save losses
    np.savez(os.path.join(save_folder, f"{args.condition}_losses.npz"), 
            train_loss=losses["train"], 
            val_los=losses["valid"])


   # Test Model
    test_loss, latent_data, test_xhat = evaluate_model(model,
                                                    test_dataloader,
                                                    loss_function,
                                                    hparams['beta'],
                                                    device)

    print(f"Total test Loss: {test_loss['total']:.3f} ; "
        f"{hparams['loss_function']['reconstruction']}: {test_loss['recon']:.3f} ; "
        f"kl: {test_loss['kl']:.3f} ")

    test_z = latent_data["z"]
    z_mu = latent_data["mu"]
    z_log_var = latent_data["log_var"] 

    np.savez(os.path.join(save_folder, f"{args.condition}_test_loss.npz"), 
            total_loss=test_loss["total"], 
            recon_loss=test_loss["recon"], 
            kl_loss=test_loss["kl"])
    print("Test Loss saved...\n")
    
    np.savez(os.path.join(save_folder, f"{args.condition}_latent_space.npz"), 
             z=test_z, 
             mu=z_mu, 
             log_var=z_log_var)
    print("Latent space saved...")

    test_xhat = data_denormalization(test_xhat, max_value, min_value)
    test_xhat = np.reshape(test_xhat, (n_frames, n_atoms, n_xyz))
    test_xhat_traj = md.Trajectory(test_xhat, traj.topology)
    test_xhat_traj.save_xtc(os.path.join(save_folder, f"{args.condition}_recon.xtc"))
    print("Reconstructed trajectory saved...")


    for i in range(hparams["latent_dim"]):
        z_variables = np.zeros_like(test_z)
        z_variables[:,i] = np.linspace(test_z[:,i].min()-0.5, test_z[:,i].max()+0.5, len(test_z))

        z_i = torch.FloatTensor(z_variables).to(device)
        xhat = model.decoder(z_i)

        xhat = xhat.detach().cpu().numpy()
        xhat = data_denormalization(xhat, max_value, min_value)
        xhat = np.reshape(xhat, (len(xhat), -1, 3))

        traj_z = md.Trajectory(xhat, traj.topology)
        traj_z[0].save_pdb(os.path.join(save_folder, f"{args.condition}_z{i+1}_first.pdb"))
        traj_z[-1].save_pdb(os.path.join(save_folder, f"{args.condition}_z{i+1}_last.pdb"))
        traj_z.save_xtc(os.path.join(save_folder, f"{args.condition}_z{i+1}.xtc"))
    
    print("Latent space trajectories saved...\n")
    print("Done!")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    # Data
    argparser.add_argument('--pdb_path', type=str) 
    argparser.add_argument('--xtc_path', type=str)
    argparser.add_argument('--condition', type=str, help='i.e. WT_apo_ChainsA_CA')
    # Fitting
    argparser.add_argument('--fit_traj', action='store_true', default=False)
    argparser.add_argument('--ref_pdb', type=str, default=None)
    argparser.add_argument('--sfactor', type=float, default=5.0)
    # Training
    argparser.add_argument('--hyperparameters', type=str, default='arquitecture1.json')
    argparser.add_argument('--load_model', type=str, default=None)
    argparser.add_argument('--train_model', action='store_true', default=False)
    argparser.add_argument('--seed', type=int, default=42)
    args = argparser.parse_args()

    main(args)