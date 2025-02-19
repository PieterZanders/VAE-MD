import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, nlayers, latent_dim, delta, dropout=0.0, negative_slope=0.0):
        super(Encoder, self).__init__()
        layers = []
        nunits = input_dim

        if isinstance(nlayers, int):
            for _ in range(nlayers):
                layers.append(torch.nn.Linear(nunits, nunits - delta))
                layers.append(torch.nn.LeakyReLU(negative_slope))
                layers.append(torch.nn.Dropout(dropout))
                nunits -= delta
        elif isinstance(nlayers, list):
            for layer_dim in nlayers:
                layers.append(torch.nn.Linear(nunits, layer_dim))
                layers.append(torch.nn.LeakyReLU(negative_slope))
                layers.append(torch.nn.Dropout(dropout))
                nunits = layer_dim

        self.encoder = torch.nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(nunits, latent_dim)
        self.fc_log_var = torch.nn.Linear(nunits, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, nlayers, output_dim, delta, dropout=0.0, negative_slope=0.0):
        super(Decoder, self).__init__()
        layers = []
        nunits = latent_dim

        if isinstance(nlayers, int):
            for _ in range(nlayers):
                layers.append(torch.nn.Linear(nunits, nunits + delta))
                layers.append(torch.nn.LeakyReLU(negative_slope))
                layers.append(torch.nn.Dropout(dropout))
                nunits += delta
        elif isinstance(nlayers, list):
            nlayers = nlayers[::-1]
            for layer_dim in nlayers:
                layers.append(torch.nn.Linear(nunits, layer_dim))
                layers.append(torch.nn.LeakyReLU(negative_slope))
                layers.append(torch.nn.Dropout(dropout))
                nunits = layer_dim

        layers.append(torch.nn.Linear(nunits, output_dim))
        layers.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class VAE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 nlayers,
                 latent_dim,
                 dropout,
                 neg_slope):

        super(VAE, self).__init__()
        self.nlayers = nlayers
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if isinstance(nlayers, int):
            nlayers = nlayers - 1
            delta = int((input_dim - latent_dim) / (nlayers + 1))
        elif isinstance(nlayers, list):
            delta = None

        self.encoder = Encoder(input_dim,
                               nlayers,
                               latent_dim,
                               delta,
                               dropout,
                               neg_slope)

        self.decoder = Decoder(latent_dim,
                               nlayers,
                               input_dim,
                               delta,
                               dropout,
                               neg_slope)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, z, mu, log_var

class VAELoss(torch.nn.Module):
    def __init__(self, loss_type='mse', reduction='sum'):
        super(VAELoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, recon_x, x, mu, log_var):
        if self.loss_type == 'bce':
            recon_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction=self.reduction)
        elif self.loss_type == 'mse':
            recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction=self.reduction)

        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_divergence