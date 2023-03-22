import os 

# TODO
# - Make a central database in mongo DB 

class RewardsWorkFlow:
    def __init__(self) -> None:
        ... 
    
    def select_train_environment(self, env_name = "car-race", sub_env = None) -> None:
        """
        env_name : Car Race or Snake 
        sub_env : Track type (example: Car track type 1/2/3)
        """
        ... 
    
    def create_model(self, parameters):
        """
        User basically makes a torch.nn.Module model which is either prebuilt or custom 
        and once created it will save the model and the parameters and all other info:
        {
            date and time, 
            trial_number, 
            model_name, 
            model_type, 
            model_parameters, 
            model_state, 
        } in a json. This will return a torch.nn.Model model instance
        """
        ... 
    
    def begin_training(self):
        ... 
    
    def push_model(self, aws_key : str):
        """
        This requires a login and also the AWS full access key to the specific folders with a size limit. This will also save 
        the number of times we are pushing the model
        """
        ... 
    
    
    
