1. generate channel model for training and testing:
   - Statics model will use file ChannelData_for_ResCNN.m
   - Mixed model will use file ChannelData_for_ResCNN_mixed.m, modify model range based on multipath value
2. training model
   - using file (for_window)channel_denoise_ResCNN.py will take input as channel, and output as deviation between input and output training
   - using file (for_window)channel_denoise_ResCNN_direct_as_output.py will take input as channel, output will be training output
3. testing model to see error deviation

#### For better training and testing version, please check version in ./adjustable code #######
