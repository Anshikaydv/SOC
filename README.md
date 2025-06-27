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
Contains fundamental Python programs:
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

- `functions.py` – Defining and calling functions, scope, return values
# 1. Basic Function (No parameters)
    def greet()
        print("Hello! Welcome to GitHUb.")
    
    greet()  # calling the function


# 2. Function with Parameters
    def add(a, b):
        result = a + b
        print(f"Sum of {a} and {b} is {result}")
    
    add(3, 7)
    

# 3. Function with Return Value
    def multiply(x, y):
        return x * y
    
    product = multiply(4, 5)
    print("Multiplication Result:", product)
    

# 4. Function with Default Arguments
    def greet_user(name="Guest"):
        print(f"Hello, {name}!")
    
    greet_user("Anshika")
    greet_user()  # uses default
    

# 5. Variable Scope: Local vs Global
    total = 0  # global variable
    
    def calculate_sum(a, b):
        total = a + b  # local variable
        print("Inside function, total =", total)
    
    calculate_sum(5, 10)
    print("Outside function, total =", total)  # global remains unchanged


# 6. Using global keyword (if needed)
    counter = 0
    
    def increment_counter():
        global counter
        counter += 1
    
    increment_counter()
    print("Counter after increment:", counter)
    
### 2. `modules/`
Covers built-in and custom Python modules:
- `math_module.py` – math functions: sqrt, ceil, floor, etc.
# math_module.py
    import math

# 1. Square root
    num = 16
    print(f"Square root of {num} is {math.sqrt(num)}")

# 2. Ceiling and Floor
    x = 5.7
    print(f"Ceiling of {x} is {math.ceil(x)}")   # rounds up
    print(f"Floor of {x} is {math.floor(x)}")   # rounds down

# 3. Power and exponential
    print(f"2 raised to power 3 is {math.pow(2, 3)}")  # 2^3 = 8.0
    print(f"Exponential of 2 is {math.exp(2)}")       # e^2

# 4. Logarithm
    print(f"Natural log of 10 is {math.log(10)}")          # base e
    print(f"Log base 10 of 1000 is {math.log10(1000)}")    # base 10

# 5. Trigonometric functions
    angle = math.radians(30)  # convert degrees to radians
    print(f"Sine of 30° is {math.sin(angle)}")
    print(f"Cosine of 30° is {math.cos(angle)}")

# 6. Constants
    print(f"Value of π (pi): {math.pi}")
    print(f"Value of e: {math.e}")

- `datetime_module.py` – working with dates and times
# Working with dates and times in Python

    import datetime

# 1. Current date and time
    now = datetime.datetime.now()
    print("Current Date and Time:", now)

# 2. Current date only
    today = datetime.date.today()
    print("Today's Date:", today)

# 3. Create a specific date
    independence_day = datetime.date(1947, 8, 15)
    print("India's Independence Day:", independence_day)

# 4. Date formatting
    print("Formatted Date:", now.strftime("%d-%m-%Y %H:%M:%S"))

# 5. Date components
    print("Year:", now.year)
    print("Month:", now.month)
    print("Day:", now.day)
    print("Hour:", now.hour)
    print("Minute:", now.minute)

# 6. Adding or subtracting time using timedelta
    one_week = datetime.timedelta(weeks=1)
    next_week = today + one_week
    print("Date after one week:", next_week)

    yesterday = today - datetime.timedelta(days=1)
    print("Yesterday was:", yesterday)

# 7. Difference between two dates
    birthday = datetime.date(2025, 1, 1)
    days_left = birthday - today
    print("Days until New Year 2025:", days_left.days)

- `numpy_module_demo.py` – Basic operations using NumPy
# Basic operations using NumPy

    import numpy as np

# 1. Create arrays
    arr1 = np.array([1, 2, 3, 4])
    print("1D Array:", arr1)
    
    arr2 = np.array([[1, 2], [3, 4]])
    print("2D Array:", arr2)

# 2. Array operations
    print("Element-wise addition:", arr1 + 5)
    print("Sum of arr1:", np.sum(arr1))
    print("Mean of arr1:", np.mean(arr1))
    print("Max of arr2:", np.max(arr2))
    
# 3. Array slicing
    print("First 2 elements of arr1:", arr1[:2])

# 4. Shape and reshape
    print("Shape of arr2:", arr2.shape)
    reshaped = arr1.reshape(2, 2)
    print("Reshaped arr1:", reshaped)
# some more numpy codes
#1.

    from array import *
    vals = array('i',[1,2,-3,4,5])
    print(vals)
    newArr = array(vals.typecode, (a*a for a in vals))
    
    print(vals.buffer_info())
    
    print(vals.typecode)
    
    for i in range(len(vals)):
        print(vals[i])
    
    print()
    
    for e in newArr:
        print(e)
    
    vals.reverse()
    print(vals)
#2.

    from array import *
    arr = array('i' , [])
    
    n = int(input("Enter the length of the array: "))
    for i in range(n):
        x = int(input("Enter the number: "))
        arr.append(x)
    
    print(arr)
    
    val = int(input("Enter the value for search: "))
    k = 0
    for e in arr:
        if e == val:
            print(k)
            break
    
        k+=1
#3.

    from numpy import *
    arr = array([1,2,3.0,5,4])
    print(arr)
    print(arr.dtype)
    
    print()
    
    arr = linspace(0,15,20)
    print(arr)
    
    print()
    
    arr = arange(1,15,4)
    print(arr)
    
    print()
    
    arr = logspace(1,40,5)
    print(arr)
    print('%.2f'%arr[0])
    print('%.2f'%arr[1])
#4.

    from numpy import *
    arr1 = array([1,2,3,4,5])
    arr2 = arr1
    arr3 = arr1.view()
    arr4 = arr1.copy()
    
    arr1[1] = 7
    print(arr1)
    print(arr2)
    print(arr3)
    print(arr4)
    print()
    print(id(arr1))
    print(id(arr2))
    print(id(arr3))
    print(id(arr4))


- `pandas_module_demo.py` – Basics of DataFrames and Series using pandas.
# pandas_demo.py

    import pandas as pd

# 1. Create a Series
    marks = pd.Series([85, 90, 95], index=['Math', 'Science', 'English'])
    print("Marks Series:", marks)

# 2. Create a DataFrame
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [20, 21, 19],
        'Branch': ['CS', 'EE', 'ME']
    }
    df = pd.DataFrame(data)
    print("Student DataFrame:", df)

# 3. Accessing data
    print("Names column:", df['Name'])
    print("First row:", df.iloc[0])

# 4. Basic operations
    print("Average age:", df['Age'].mean())

- `matplotlib_module_demo.py` – Plotting with Matplotlib
# matplotlib_demo.py

    import matplotlib.pyplot as plt

# 1. Line plot
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 25, 30, 35]
    plt.plot(x, y, label="Line Plot", color='green')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Line Graph Example")
    plt.legend()
    plt.grid(True)
    plt.show()

# 2. Bar plot
    subjects = ['Math', 'Science', 'English']
    scores = [85, 90, 80]
    plt.bar(subjects, scores, color='skyblue')
    plt.title("Subject Scores")
    plt.ylabel("Marks")
    plt.show()

# 3. Pie chart
    labels = ['Python', 'C++', 'Java']
    sizes = [40, 35, 25]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title("Programming Language Usage")
    plt.show()

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


