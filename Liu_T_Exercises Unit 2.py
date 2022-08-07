# Full Spectrum Bioinformatics Exercises Unit 2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Exploring Python: Exercise 3
    # If you started with 3 bacteria instead of 1, how would that change the number of bacteria present after 1 day?.

    division_time = 2  # Division time in hours
    starting_bacteria = 3
    offspring_per_division = 2  # Bacteria produce 2 offspring per division
    total_hours = 2 * 7 * 24  # Weeks * # days in week * Hours in Day
    total_divisions = total_hours / division_time
    total_bacteria = starting_bacteria * offspring_per_division ** total_divisions

    print("Total bacteria: ", total_bacteria)
    print()
    # Exploring Python: Exercise 4 Let's imagine that a tree starts from a seed that is roughly 1 cm tall, and grows at a fixed rate of
    # 1 m per year. Write python code that will calculate how big the tree is after 7 years. Hint: you can use the math
    # operations described above to write the equation for a line (y = mx + b), with the number
    # of years being the x variable.

    seed_start_height = 0.01  # m or 1 cm
    growth_rate = 1  # m per year
    time = 7  # years

    y = (growth_rate * time) + 0.01

    print(f"Height after {time} years = {y}")
    print()

    # Representing ... with Python Strings: Exercise 4 Write python code that converts an RNA sequence to a DNA
    # sequence by replacing all characters representing Uracil with characters representing Thymidine

    rna_seq = 'AACGAAUUAUGUCCCUUG'
    dna_seq = rna_seq.replace('U', 'T')
    print(f'Starting Sequence: {rna_seq} \n'
          f'Resultant Sequence: {dna_seq}')
    print()

