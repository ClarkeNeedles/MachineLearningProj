import h5py
import pandas as pd

data1 = pd.read_csv('/MachineLearningProj/Data/JumpingPhoneInBackPocket.csv')
data2 = pd.read_csv('/MachineLearningProj/Data/WalkingPhoneInBackPocket.csv')
data3 = pd.read_csv('/MachineLearningProj/Data/JumpingPhoneInFrontPocket.csv')
data4 = pd.read_csv('/MachineLearningProj/Data/WalkingPhoneInFrontPocket.csv')
data5 = pd.read_csv('/MachineLearningProj/Data/JumpingPhoneInHand.csv')
data6 = pd.read_csv('/MachineLearningProj/Data/WalkingPhoneInHand.csv')

with h5py.File('project_data.h5', 'w') as h5:

    # Creating the first 3 groups
    raw_data = h5.create_group('Raw Data')
    pp_data = h5.create_group('Pre-processed Data')
    segmented_data = h5.create_group('Segmented Data')

    # Creating the subgroups under Raw Data
    raw_data.create_group('Andrew P')
    raw_data.create_group('Ben M')
    raw_data.create_group('Clarke N')

    # Creating the subgroups under Pre-processed Data
    pp_data.create_group('Andrew P')
    pp_data.create_group('Ben M')
    pp_data.create_group('Clarke N')

    # Creating the subgroups under Segmented Data
    segmented_data.create_group('Train')
    segmented_data.create_group('Test')

    # Adding the raw data
    raw_data['Andrew P'].create_dataset('JumpingPhoneInHand', data=data5.to_numpy())
    raw_data['Andrew P'].create_dataset('WalkingPhoneInHand', data=data6.to_numpy())
    raw_data['Ben M'].create_dataset('JumpingPhoneInBackPocket', data=data1.to_numpy())
    raw_data['Ben M'].create_dataset('WalkingPhoneInBackPocket', data=data2.to_numpy())
    raw_data['Clarke N'].create_dataset('JumpingPhoneInFrontPocket', data=data3.to_numpy())
    raw_data['Clarke N'].create_dataset('WalkingPhoneInFrontPocket', data=data4.to_numpy())



