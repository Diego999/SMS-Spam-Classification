from utils import get_data

if __name__ == '__main__':
    data = get_data()
    for d in data:
        print(' '.join(d['tokens']))