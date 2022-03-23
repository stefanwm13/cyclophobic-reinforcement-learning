import pickle
import argparse

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--arr',
                help='Array of integers')

    args = parser.parse_args()
    data = eval(args.arr)
    print(data)
    
    file2 = open('results/q_table_cycle_3.dat', 'rb')
    goal_q_table = pickle.load(file2)
    file2.close()

    #print(goal_q_table[str(data[0])])

    print(goal_q_table[str(data[0])][str(data[1])])
