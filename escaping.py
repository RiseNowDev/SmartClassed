import csv

input_file = 'E:/SmartClassed/data/ars_sup_item.csv'
output_file = 'E:/SmartClassed/data/ars_sup_item_cleaned.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
    
    for row in reader:
        cleaned_row = [col.replace('"', '""') if '"' in col else col for col in row]
        writer.writerow(cleaned_row)

print("CSV cleaned and saved to:", output_file)
