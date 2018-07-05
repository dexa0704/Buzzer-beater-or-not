from imageProcessing import *


if __name__ == "__main__":

    print("Welcome to buzzer beater application :)\n\n")
    success = testProgramOnAllData(rootDataSet)
    print("Program successfully solved all data set with precision of "+str(success)+" percentage correct solved shots.")

    # testingShotclockByPose()
    # hand_ball_contact = checkBallHandContact("Dejan")
    # checkOneShot('../../MyDataset/Bacanje/Poza3-Regularno/IMAG0090_BURST011.jpg')
