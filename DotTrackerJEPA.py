from abstractjepa import JEPA


class DotTrackerJEPA(JEPA):
    def __init__(self, context_encoder, predictor, loss_calculator):
        super(DotTrackerJEPA, self).__init__(context_encoder, predictor, loss_calculator)

    def encode_x(self, states):
        flattened_states = states.reshape(-1, *states.shape[2:])
        encoded_states = self.encoder(flattened_states)
        encoded_states = all_states_enc.view(*states.shape[:2], -1)
        return encoded_states

    def encode_y(self, y):
        '''y encoding explicitly ignored in this implementation - all done through self.encode_x()'''
        pass

    def predict_encoded_y(self, first_encoded_state, actions):
        predicted_encoded_states = self.predictor.predict_sequence(
            enc=first_encoded_state, actions=actions
        )
        return predicted_encoded_states

    def get_loss(self, encoded_states, predicted_encoded_states, z=None):
        variance_loss, covariance_loss, invariance_loss = self.loss_calculator['vicreg'].calculate_VICReg_loss(encoded_states, predicted_encoded_states)
        return variance_loss + covariance_loss + invariance_loss

    def forward(self, states, actions):
        encoded_states = self.encode_x(states)
        predicted_encoded_states = self.predict_encoded_y(encoded_states, actions)
        loss = self.get_loss(encoded_states, predicted_encoded_states)
        return loss