class HMM(object):
    def __init__(self):
        self.states = ['START', 'STOP']
        self.labels = []
        self.a = None 
        self.b = None
        
    def readfile(self, file):
        for line in open(file, "r"):
            if line != '\n':
                x, y = line.strip().split(' ')
                if x not in self.labels: 
                    self.labels.append(x)
                if y not in self.states:
                    self.states.append(y)
    
    def estimate_b(self, file): 
    #TODO add in the start and stop states? 
        y = []
        y_to_x = [[0 for i in range(len(self.states))] for i in range(len(self.labels))]
        for line in open(file, "r"):
            if line != '\n':
                x, y = line.split(' ')
                y_index = self.states.index(y.strip())
                x_index = self.labels.index(x.strip())
                y_to_x[x_index][y_index] += 1
        
        y =  [sum(i) for i in zip(*y_to_x)] 
        
        for i in range(len(self.labels)):
            for j in range(len(self.states)):
                if y[j] == 0: 
                    y_to_x[i][j] = 0
                else: 
                    y_to_x[i][j] = y_to_x[i][j]/y[j]
        self.b = y_to_x
        return self.b

    def improved_estimate_b(self, file, k=0.5):
        #add 'unknown' label 
        self.labels.append("#UNK#")

        y = []
        y_to_x = [[0 for i in range(len(self.states))] for i in range(len(self.labels))]
        for line in open(file, "r"):
            if line != '\n':
                x, y = line.split(' ')
                y_index = self.states.index(y.strip())
                x_index = self.labels.index(x.strip())
                y_to_x[x_index][y_index] += 1
        
        y =  [sum(i) for i in zip(*y_to_x)] 
        unknown_x = self.labels.index("#UNK#")
        
        for i in range(len(self.labels)):
            for j in range(len(self.states)):
                if (y[j]+k) == 0: 
                    y_to_x[i][j] = 0
                if i == unknown_x: 
                    y_to_x[i][j] = k/(y[j]+k)
                else: 
                    y_to_x[i][j] = y_to_x[i][j]/(y[j]+k)
        self.b = y_to_x

        return self.b

    def get_tag(self, x): 
        if x not in self.labels:
            x = "#UNK#" 
        x_index = self.labels.index(x)
        max_y_index = self.b[x_index].index(max(self.b[x_index]))
        return self.states[max_y_index]

    def predict(self, test_file):
        f = open("dev.p2.out", "w")
        for line in open(test_file, "r"):
            if line != "\n":
                x = line.strip()
                f.write("{0} {1}\n".format(x, self.get_tag(x)))
            else: 
                f.write(line)

        f.close()

# #TODO this does not look right
#     def evaluate(self, y_pred, y_actual):
#         print(self.states)
#         confusion_matrix = [[0 for i in self.states] for i in self.states] 
#         for i in range(len(y_pred)):
#             pred_index = self.states.index(y_pred[i])
#             actual_index = self.states.index(y_actual[i])
#             confusion_matrix[actual_index][pred_index] += 1
        
#         true_positive = [confusion_matrix[i][i] for i in range(len(self.states))]
#         false_positive = [0 for i in range(len(self.states))]
#         false_negative = [0 for i in range(len(self.states))]
#         precision = recall = [0 for i in range(len(self.states))]

#         for correct in range(len(self.states)):
#             for i in range(len(self.states)):
#                 if i != correct: 
#                     false_positive[correct] += confusion_matrix[i][correct]
        
#         for correct in range(len(self.states)):
#             for i in range(len(self.states)):
#                 if i != correct: 
#                     false_negative[correct] += confusion_matrix[correct][i]


#         for i in range(len(true_positive)):
#             if (true_positive[i] + false_positive[i]) == 0: 
#                 precision[i] = 0
#             else: 
#                 precision[i] = true_positive[i]/(true_positive[i] + false_positive[i])
        
#         precision = sum(precision)

#         for i in range(len(true_positive)):
#             if (true_positive[i] + false_negative[i]) == 0: 
#                 recall[i] = 0
#             else: 
#                 recall[i] = true_positive[i]/(true_positive[i] + false_negative[i])
#         recall = sum(recall)
        
#         f1 = 2/((1/precision) + 1/recall)
#         print("Precision: ", precision)
#         print("Recall: ", recall)
#         print("F1 score: ", f1)
#         print(confusion_matrix[3])


model = HMM()
model.readfile("./EN/train")
y_to_x = model.improved_estimate_b("./EN/train")
x_test = []
y_actual = []


for line in open("./EN/dev.out", "r"):
    if line != "\n":
        x, y = line.strip().split(" ")
        y_actual.append(y)
model.predict("./EN/dev.in")
# model.evaluate(y_pred, y_actual)