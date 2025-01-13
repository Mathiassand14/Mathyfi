import csv
import re

def extract_latex_commands(csv_filename):
    
    unique_commands = set()

    # Stop karakterer. Vi kan tilføje / fjerne nogle af dem, hvis vi vil.
    token_pattern = re.compile(r'^([^\s\{\}\(\)\[\]\^_]+)')

    with open(csv_filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            latex_code = row['latex']
            # splitter den efter '\'
            parts = latex_code.split('\\')
            # parts[0] er alting før den første '\'; Det ignorerer vi
            for chunk in parts[1:]:
                # Leder efter en match i starten af chunk
                match = token_pattern.match(chunk)
                if match:
                    command = match.group(1)
                    unique_commands.add(command)

    # Returner den sorterede liste af unikke kommandoer i latex koden
    return sorted(unique_commands)

if __name__ == "__main__":
    csv_filename = "mathwriting-2024.csv"
    commands = extract_latex_commands(csv_filename)
    print("Unique LaTeX-like tokens after backslashes:")
    print(f"Total: {len(commands)}")
    for cmd in commands:
        print(cmd)
