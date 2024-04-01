import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Generator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1).squeeze(1)


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        generated_image = self.generator(x)
        discriminator_result = self.discriminator(generated_image)
        return discriminator_result

    def train_model(self, source_loader, target_loader, device, num_epochs=10, lr_G=0.001, lr_D=0.001):
        """
        Entraîne un modèle GAN pour l'adaptation de domaine.
        """
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D)
        criterion = nn.BCELoss()
        self.train()

        for epoch in range(num_epochs):
            for source_data, target_data in zip(source_loader, target_loader):
                source_images, _ = source_data
                target_images, _ = target_data
                source_images = source_images.to(device)
                target_images = target_images.to(device)

                # Entraîner le discriminateur sur les vraies images cibles
                optimizer_D.zero_grad()
                real_preds = self.discriminator(target_images)
                real_loss = criterion(real_preds, torch.ones_like(real_preds))

                # Entraîner le discriminateur sur les fausses images générées à partir des images source
                fake_images = self.generator(source_images)
                fake_preds = self.discriminator(fake_images.detach())
                fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                # Entraîner le générateur pour tromper le discriminateur
                optimizer_G.zero_grad()
                tricked_preds = self.discriminator(fake_images)
                g_loss = criterion(tricked_preds, torch.ones_like(tricked_preds))

                g_loss.backward()
                optimizer_G.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

    def transform_images(self, source_loader):
        transformed_images = []
        labels_list = []
        self.generator.eval()
        with torch.no_grad():
            for images, labels in source_loader:
                images = images.to(self.device)
                transformed = self.generator(images).cpu()
                transformed_images.append(transformed)
                labels_list.append(labels)
        self.generator.train()
        transformed_dataset = TensorDataset(torch.cat(transformed_images, dim=0), torch.cat(labels_list, dim=0))
        transformed_loader = DataLoader(transformed_dataset, batch_size=64, shuffle=True)
        return transformed_loader


class DigitClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64*8*8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def train_model(self, classifier, data_loader, device, epochs=10):
        classifier.train()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')