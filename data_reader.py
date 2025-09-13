import os
import pandas as pd
import matplotlib.pyplot as plt

class DataReader:
    def __init__(self, viz_data=False):

        try:
            # samsung/dataset/for_my_learning_purpose-dataset_collection/fashion_mnist
            # DATASET_PATH_SAMSUNG = "/samsung/dataset/for_my_learning_purpose_dataset_collection/fashion_mnist"
            # self.dataset_path_samsung = os.path.join("/", "samsung", "dataset", 
            #                         "for_my_learning_purpose_dataset_collection", 
            #                         "fashion_mnist")

            self.dataset_path = os.path.join("./dataset")
            self.df = pd.read_csv(os.path.join(self.dataset_path, "fashion-mnist_train.csv"))   # train_dataset name=fashion-mnist_train.csv
            # print(self.df.head(3))
            # print(self.df.shape)

            if viz_data:
                self.visualize_data()
            

        except FileExistsError as e:
            print("Error:", e)

        except pd.errors.ParserError as e:
            print("Parsing error while reading CSV:", e)

        except Exception as e:
            print("Unexpected error:", e)

    def visualize_data(self):
        img = self.df.iloc[0, 1:].values.reshape(28, 28)
        # print(img)

        plt.imshow(img)
        plt.show()

# if __name__ == "__main__":
#     data_reader = DataReader(viz_data=False)
    # data_reader = DataReader(viz_data=True)