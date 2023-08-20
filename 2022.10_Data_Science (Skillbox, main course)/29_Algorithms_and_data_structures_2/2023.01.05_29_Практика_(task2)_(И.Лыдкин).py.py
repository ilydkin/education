
# Функция принимает на вход строку, которая
# состоит из скобок трех типов: круглые, квадратные
# и фигурные. Функция должна проверить, является ли
# переданная последовательность скобок сбалансированной
# или нет. Функция возвращает True, если последовательность
# сбалансирована, и False – в противном случае.

def is_balanced(line):
    pairs = {')':'(', ']':'[', '}':'{'}
    stack = []
    for i in line:
        if i=='(' or i=='[' or i=='{':
            stack.append(i)
        else:
            if pairs[i] != stack.pop():
                return False
    if len(stack) == 0:
        return True
    else:
        return False


def test_is_balanced():
    cases = {
        '((((((((())))))))': False,
        '{[()]}{{}}': True,
        '{[()]}{]{}}': False,
        '{{{((([[[]]])))}}}': True,
        '{}': True,
        '(': False,
        '(}': False,
        '(((())))[]{}': True,
        '((()': False,
        '[{}{})(]': False,
        '{[]{([[[[[[]]]]]])}}': True,
        '{[]{([[[[[[]])]]])}}': False,
    }
    for i, case in enumerate(cases.keys()):
        if is_balanced(case) == cases[case]:
            print(f'{i}: OK')
        else:
            print(f'{i}: Not OK')


def main():
    test_is_balanced()


if __name__ == '__main__':
    main()