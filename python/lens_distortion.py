
# We will interact with the C++ using the subprocess call, so we need to import the subprocess module
import subprocess
from pathlib import Path
import shutil

# Define the path to the C++ executable
_cpp_executable = Path('../build/cameraLensCalibration').resolve()

# Define the path to the test image
test_image = Path('../example/building.png').resolve()

# Default parameter string
default_params = "0.8 0.0 3.0 3.0 10.0 div True"


def call_cpp(test_image: Path, output_dir: Path, params: str = default_params) -> str:
    # Delete contents of the output directory if it exists, otherwise create it
    for file in output_dir.glob('*'):
        file.unlink()
    # Check that the C++ executable exists
    if not _cpp_executable.exists():
        raise FileNotFoundError(f'C++ executable not found at {_cpp_executable}')
    # Check that the test image exists
    if not test_image.exists():
        raise FileNotFoundError(f'Test image not found at {test_image}')
    # Copy the test image to the output directory, not move it
    shutil.copy(test_image, output_dir)
    # Construct the full string to call the C++ executables
    im = test_image.parent.resolve()
    call_string = f'{_cpp_executable} {output_dir.resolve()}/ {output_dir.resolve()}/ {params}'
    # Call the C++ executable with the call string
    subprocess.run(call_string, shell=True, capture_output=False, check=True)
    print(call_string)
    # Check if there is a file called 'output.txt' in the output directory
    output_file = Path(output_dir / 'output.txt').resolve()
    if not output_file.exists():
        raise FileNotFoundError(f'Output file not found at {output_file}')
    # Read the output file
    with open(output_file, 'r') as f:
        result = f.read()
    return result
    

if __name__ == '__main__':
    # Define the output directory
    output_dir = Path('../output')
    # Make the output directory if it does not exist
    output_dir.mkdir(exist_ok=True)
    # Call the C++ executable
    result = call_cpp(test_image, output_dir)
    print(result)
