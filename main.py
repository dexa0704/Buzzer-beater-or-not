from imageProcessing import *


if __name__ == "__main__":

    print("Welcome to buzzer beater application :)\n\nPerforming training with 70% of data set ...\n\n")
    train_success = testProgramOnAllData(rootDataSet, "train")
    print("Training successfully finished. Solved training data set with precision of " +
          str(train_success) + " percentage correct solved shots.")

    print("Performing testing application for other 30% data...\n\n")
    test_success = testProgramOnAllData(rootDataSet, "test")
    print("Program successfully solved test data set with precision of " +
          str(test_success) + " percentage correct solved shots.")

    # testingShotclockByPose()
    # hand_ball_contact = checkBallHandContact("Dejan")
    # checkOneShot('../../MyDataset/Bacanje/Poza3-Regularno/IMAG0090_BURST011.jpg')
