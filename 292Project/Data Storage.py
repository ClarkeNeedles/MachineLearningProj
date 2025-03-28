import h5py
import pandas as pd

df = pd.read_csv()

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

