import csv
import os
def txt_to_csv(inputPath, inputName, outputPath, outputName):
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    outputName = os.path.join(outputPath, outputName)
    csvFile = open(outputName, 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)

    f = open(os.path.join(inputPath, inputName), 'r', encoding='utf-8')
    for line in f:
        csvRow = line.split(',')
        writer.writerow(csvRow)

    f.close()
    csvFile.close()