from DotTrackerJEPA import DotTrackerJEPA
from models import MeNet5, RNNPredictor
from lossCalculators import VICRegLossDotCalculator
import torch.nn as nn

if __name__ == "__main__":

    encoder = MeNet5(embedding_dimension=512, input_channels=2, width_factor=1)
    predictor = RNNPredictor(
        hidden_size=encoder.embedding_dimension,
        num_layers=1,
        action_dimension=2,
    )
    expander = nn.Identity()
    loss_calculators = {'vicreg': VICRegLossDotCalculator(expander)}
    dtj = DotTrackerJEPA(encoder, predictor, loss_calculators)
