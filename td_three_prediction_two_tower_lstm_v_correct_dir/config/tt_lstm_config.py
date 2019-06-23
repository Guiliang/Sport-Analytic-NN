import yaml
from td_three_prediction_two_tower_lstm_v_correct_dir.support.config import InitWithDict


class TTLSTMCongfig(object):

    def __init__(self, init):
        self.learn = TTLSTMCongfig.learn(init["learn"])
        self.Arch = TTLSTMCongfig.Arch(init["Arch"])

    class learn(InitWithDict):
        model_type = None
        max_trace_length = None
        feature_number = None
        batch_size = None
        gamma = None
        output_layer_size = None
        dense_layer_num = None
        h_size = None
        embed_size = None
        dropout_keep_prob = None
        use_hidden_state = None
        model_train_continue = None
        scale = None
        feature_type = None
        iterate_num = None
        learning_rate = None
        sport = None
        reward_type = None
        save_mother_dir = None
        if_correct_velocity = None
        apply_softmax = None
        merge_tower = None

    class Arch(InitWithDict):

        def __init__(self, init):
            # super(TTLSTMCongfig.Arch, self).__init__(init)
            self.HomeTower = TTLSTMCongfig.Arch.HomeTower(init["HomeTower"])
            self.AwayTower = TTLSTMCongfig.Arch.AwayTower(init["AwayTower"])

        class HomeTower(InitWithDict):
            lstm_layer_num = None
            home_h_size = None

        class AwayTower(InitWithDict):
            lstm_layer_num = None
            away_h_size = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return TTLSTMCongfig(config)
