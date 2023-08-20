import random

# Функция bubble_sort принимает список целых чисел
# data и сортирует его в порядке убывания элементов с
# помощью пузырьковой сортировки. Кроме того, функция
# должна посчитать количество операций (итераций цикла),
# которые выполняет алгоритм, и вернуть это число вызывающей
# стороне.

def bubble_sort(data):
    iterations = 0
    for i in range(len(data)-1):
        for j in range(len(data) - i - 1):
            iterations += 1
            if data[j] < data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return iterations


def test_sorted():
    data = [random.randint(0, 1000) for i in range(100)]
    data_to_sort = data.copy()
    bubble_sort(data_to_sort)
    if data_to_sort == sorted(data, reverse=True):
        print('OK')
    else:
        print('NOT OK')


def make_observations():
    size = 10
    results = []
    for i in range(100):
        data = [random.randint(0, 1000) for i in range(size)]
        results.append((size, bubble_sort(data)))
        size += 10
    return results


def main():
    test_sorted()
    with open('bubble.csv', 'w') as file:
        file.write(f'size, iterations\n')
        for row in make_observations():
            file.write(f'{row[0]}, {row[1]}\n')
    print('Done!')


if __name__ == '__main__':
    main()