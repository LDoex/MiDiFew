import csv
import os
def txt_to_csv(filePath, outputPath):
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    outputName = os.path.join(outputPath, 'NSL_KDD.csv')
    csvFile = open(outputName, 'a', newline='', encoding='utf-8')