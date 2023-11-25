import subprocess
import re

# Define the different configurations
ghost_types = ['afraid', 'fearless', 'terrified']
layouts = ['large_filter', 'large_filter_walls']
num_ghosts_options = [1, 2, 3, 4, 5]

# Open a file to write the results
with open('game_results.txt', 'w') as results_file:
    # Iterate over all combinations of configurations
    for ghost_type in ghost_types:
        for layout in layouts:
            for num_ghosts in num_ghosts_options:
                # Construct the command
                command = f'python run.py --ghost {ghost_type} --nghosts {num_ghosts} --visible --layout {layout} --seed 42'
                print(command)

                # Write the configuration title
                config_title = f'Configuration: Ghost Type: {ghost_type}, Number of Ghosts: {num_ghosts}, Layout: {layout}'
                results_file.write(config_title + '\n')
                results_file.write(f'Running command: {command}\n')

                # Run the command
                try:
                    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                    output = output.decode('utf-8', errors='replace')  # Decode output to string with error handling

                    # Use regular expressions to find the score and time in the output
                    score_match = re.search(r'Score: (-?\d+)', output)
                    time_match = re.search(r'Computation time: ([\d.]+)', output)

                    # Write the score and time to the results file
                    if score_match:
                        score = int(score_match.group(1))
                        results_file.write(f'Score: {score}\n')
                    else:
                        results_file.write('No score found in the output.\n')

                    if time_match:
                        time = float(time_match.group(1))
                        results_file.write(f'Computation time: {time}\n')
                    else:
                        results_file.write('No computation time found in the output.\n')

                except subprocess.CalledProcessError as e:
                    results_file.write(f'An error occurred: {e.output}\n')

                results_file.write('\n')  # Add a newline for readability between runs

# Print out the path to the results file
print("Results have been written to game_results.txt")
