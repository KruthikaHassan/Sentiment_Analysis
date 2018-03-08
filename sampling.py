


import csv
import random


csv_file = open('SAD.csv')


text = []
labels = []

pos_indx = []
neg_indx =[]
indx = 0
for row in csv.reader(csv_file):

    if indx == 0:
        indx += 1
        continue

    line = row[-1]
    label = int(row[1])
    text.append(line)
    labels.append(label)

    if label == 1:
        pos_indx.append(indx-1)
    else:
        neg_indx.append(indx-1)
    
    indx += 1

print(len(pos_indx))
print(len(neg_indx))

random.shuffle(neg_indx)
random.shuffle(pos_indx)

print("shuffled")

total_records = 50000

out_file = open('sampling.csv','w')
out_file.write("Labels, Text")
for num_rec in range(len(labels)):

    pin = pos_indx.pop(0)
    nin = neg_indx.pop(0)

    out_file.write("%d, %s\n" % (labels[pin], text[pin]))
    out_file.write("%d, %s\n" % (labels[nin], text[nin]))

    total_records -= 1

    if total_records == 0:
        break

close(out_file)


