import pkg_resources
import os

def track(name=None):
    # Get the current module's name dynamically
    package_name = __name__.split('.')[0]  # 'racelab' if the file is in 'racelab.deepracer'
    
    # Get the path to the 'data' folder in the current package (e.g., 'racelab.deepracer/data')
    data_folder = pkg_resources.resource_filename(package_name, 'deepracer/data')
    
    if name is None:
        # List all .npy files in the 'data' folder and strip the '.npy' extension
        track_names = [os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith('.npy')]
        
        # Print available track names
        print("Available track names:", track_names)
    else:
        # Return the path for the specific track if a name is provided
        return pkg_resources.resource_filename(package_name, f'deepracer/data/{name}.npy')
