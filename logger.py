import csv

class Logger:
    def log(self, filename, steps, reward):
        with open(filename, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([steps, reward])
