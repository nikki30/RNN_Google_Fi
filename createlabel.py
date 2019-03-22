import csv

def great(row):
    if (row[1] > row[2]) and (row[1] > row[3]):
        return 0
    elif (row[2] > row[1]) and (row[2] > row[3]):
        return 1
    else:
        return 2


with open('D:\q2\CS219\Project\py-code\carrier_data_kehk_1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    final = []
    for row in csv_reader:
        if line_count == 0:
            print(row)
            row.append('carrier')
            final.append(row)
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            row.append(great(row))
            final.append(row)
            line_count += 1
    print(f'Processed {line_count} lines.')
    print(final)

with open('D:\q2\CS219\Project\py-code\carrier.csv', mode='w', newline='') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for row in final:
        employee_writer.writerow(row)
    


