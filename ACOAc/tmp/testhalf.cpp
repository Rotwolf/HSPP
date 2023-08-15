#include <iostream>

using namespace std;

bool canBecomeZero(double value) {
    while (value != 0.0) {
        value = 0.5 * value;
        cout << value << endl;
        if (value == 0.0) {
            return true;
        }
    }
    return false;
}

int main() {
    double initialValue;

    cout << "Enter a double value: ";
    cin >> initialValue;

    if (canBecomeZero(initialValue)) {
        cout << "The value can become zero through repeated halving." << endl;
    } else {
        cout << "The value cannot become zero through repeated halving." << endl;
    }
   
    return 0;
}