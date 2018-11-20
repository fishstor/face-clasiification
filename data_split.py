import csv

def split_data(filename):
    '''
    spliting whole datasets into three files for train, test and validation

    params
        filename: the full datasets path for spliting
    '''

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    
        train_data = [row[:-1] for row in rows if row[-1] == 'Training']
        with open('datasets/train.csv', 'w+', newline='') as f1:
            writer1 = csv.writer(f1)
            for data in train_data:
                writer1.writerow(data)
    
        public_test = [row[:-1] for row in rows if row[-1] == 'PublicTest']
        with open('datasets/validation.csv', 'w+', newline='') as f2:
            writer2 = csv.writer(f2)
            for data in public_test:
                writer2.writerow(data)

        private_test = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        with open('datasets/test.csv', 'w+', newline='') as f3:
            writer3 = csv.writer(f3)
            for data in private_test:
                writer3.writerow(data)


if __name__ == '__main__':
    split_data('fer2013/fer2013.csv')
