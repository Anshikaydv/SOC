# SOC
# Python Practice Repository
This repository contains my practice code while learning the basics of Python, including core syntax, built-in modules, and other beginner-friendly topics.
---

python-practice/
├── 01_basics/
│   ├── variables.py
│   ├── datatypes.py
│   └── loops.py
|
├── 02_functions_modules/
│   ├── functions.py
│   ├── built_in_modules.py
│   └── custom_modules.py
|
├── 03_file_handling/
│   └── file_read_write.py
|
├── 04_advanced_topics/
│   ├── list_comprehensions.py
│   ├── lambda_map_filter.py
│   └── error_handling.py
|
├── resources/
│   ├── youtube_links.md
│   └── cheat_sheets.md
|
├── README.md
└── .gitignore

# 🐍 Python Practice Repository

This repository contains my practice code while learning the basics of Python, including core syntax, built-in modules, and other beginner-friendly topics.

---

## 🧑‍💻 Sections

### `01_basics/`
- `variables.py` – Demonstration of variable declaration, input/output, and basic data 

# Variable Declaration
    name = "Anshika"
    age = 21
    height = 5.4
    is_student = True

# Printing variables
    print("Name:", name)
    print("Age:", age)
    print("Height:", height)
    print("Student:", is_student)
    
 # Getting input from user
    your_name = input("Enter your name: ")
    your_age = input("Enter your age: ")  # Note: input returns string

# Type Casting
    your_age = int(your_age)

# Using variables
    print(f"Hello {your_name}, next year you will be {your_age + 1} years old.")

 # Checking variable types
    print("Type of your_name:", type(your_name))
    print("Type of your_age:", type(your_age))

 # Multiple assignment
    a, b, c = 10, 20, 30
    print("Values of a, b, c:", a, b, c)
    
# Swapping variables
    a=5
    b=6

    temp=a
    a=b
    b=temp

    print(a)
    print(b)

    a=a+b
    b=a-b
    a=a-b

    print(a)
    print(b)

    a=a^b
    b=a^b
    a=a^b

    print(a)
    print(b)

    a,b=b,a

    print(a)
    print(b)

- `datatypes.py` – Demonstration of common Python data types: strings, lists, dictionaries, sets, and tuples

# ----------------------------
# 1. STRING
# ----------------------------
    name = "Python Programming"
    print("String:", name)
    print("First 6 letters:", name[:6])
    print("Uppercase:", name.upper())
    print("Replaced:", name.replace("Python", "C++"))

# ----------------------------
# 2. LIST
# ----------------------------
    fruits = ["apple", "banana", "mango"]
     print("\nList:", fruits)
    fruits.append("orange")  # add element
    print("After appending:", fruits)
    fruits.remove("banana")  # remove element
    print("After removing banana:", fruits)
    print("Second fruit:", fruits[1])
    print("Looping through list:")
    for fruit in fruits:
        print("-", fruit)

# ----------------------------
# 3. TUPLE
# ----------------------------
    coordinates = (10, 20)
    print("\nTuple:", coordinates)
    print("X coordinate:", coordinates[0])
# coordinates[0] = 100  # This will raise an error because tuples are immutable

# ----------------------------
# 4. SET
# ----------------------------
    unique_numbers = {1, 2, 3, 2, 1}
    print("\nSet (duplicates removed):", unique_numbers)
    unique_numbers.add(4)
    print("After adding 4:", unique_numbers)
    unique_numbers.discard(2)
    print("After discarding 2:", unique_numbers)

# ----------------------------
# 5. DICTIONARY
# ----------------------------
    student = {
    "name": "Anshika",
    "age": 21,
    "branch": "Chemical Engineering"
    }
    print("\nDictionary:", student)
    print("Student Name:", student["name"])
    student["age"] = 22  # updating value
    student["college"] = "IIT Bombay"  # adding new key-value pair
    print("Updated Dictionary:", student)

# Loop through dictionary
    print("Student Details:")
    for key, value in student.items():
    print(f"{key} : {value}")

- `conditionals.py` – if, elif, else statements
# Conditions
       x = 8
      r = x % 2

    if r == 0:
        print("even")
    if (x > 5):
        print('great')
    else:
        print('not great')

    else:
        print("odd")

    print("bye")

    a = 5
    if a==1:
        print("a is 1")
    
    elif a==2:
        print("a is 2")
    
    elif a==3:
        print("a is 3")
    
    elif a==4:
        print("a is 4")
    
    else:
        print("wrong input")

- `loops.py` – for loops, while loops, nested loops
# ----------------------------
# 1.FOR LOOPS
# ----------------------------
    x = ['anshika', 65,2.5]
    b = 'Anshika'
    for i in x:
        print(i)
    
    for i in b:
        print(i)
    
    for i in [2,6,'laddoo']:
        print(i)
    
    for i in range(20,11,-2):
        print(i)

# ----------------------------
# 2.WHILE LOOPS
# ----------------------------
    av=5
    x=int(input("how many candies you want?"))
    
    i = 1
    while i <= x:
        if i>av:
            break
    
         print("candy")
        i+=1
    
    print("bye")
    
    for i in range(1,31):
        if i % 3 ==0:
            continue
    
        print(i)

# ----------------------------
# 3. NESTED LOOPS
# ----------------------------
    for i in range(4):
        for j in range(4):
            print("# ",end="")
    
        print()
    
    print()
    
    for i in range(4):
        for j in range(i+1):
            print("# ",end="")
    
        print()
    
    print()
    
    for i in range(4):
        for j in range(4-i):
            print("# ",end="")
    
        print()
    
    print()
    
    for i in range(4):
        for j in range(i, 4):
            print(j+1, end="")
        print()


### `02_functions_modules/`
- `functions.py` – Defining and calling functions
- `built_in_modules.py` – Using `math`, `random`, and `datetime`
- `custom_modules.py` – Creating and importing custom modules

### `03_file_handling/`
- `file_read_write.py` – Reading and writing to text files

### `04_advanced_topics/`
- `list_comprehensions.py` – Efficient list creation
- `lambda_map_filter.py` – Functional programming basics
- `error_handling.py` – `try`, `except`, and error types


