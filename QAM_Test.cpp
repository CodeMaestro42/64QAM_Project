#include <iostream>
#include <cmath> 
#include <random>
#include <unordered_map>
#include <string>
#include <vector>
using namespace std; 

class BitStream {
public: 
    string bitstring;

    BitStream() : bitstring("000000") {}
};

class Symbol {
public: 
    int amplitude; 
    int phase; 

    Symbol() : amplitude(0), phase(0) {}
};

void HammingCode()
{
//
}

void RealModulation(BitStream &TBits, Symbol &Sym)
{ 
    if ((TBits.bitstring[0] == '0') && (TBits.bitstring[1] == '0') && (TBits.bitstring[2] == '0'))

    {
        Sym.amplitude = -4;
    }

    else if ((TBits.bitstring[0] == '0') && (TBits.bitstring[1] == '0') && (TBits.bitstring[2] == '1'))

    {
        Sym.amplitude = -3;
    } 
    else if ((TBits.bitstring[0] == '0') && (TBits.bitstring[1] == '1') && (TBits.bitstring[2] == '1'))

    {
        Sym.amplitude = -2;
    }
    else if ((Bits.bitstring[0] == '0') && (Bits.bitstring[1] == '1') && (TBits.bitstring[2] == '0'))

    {
        Sym.amplitude = -1;
    }
    else if ((Bits.bitstring[0] == '1') && (TBits.bitstring[1] == '1') && (Bits.bitstring[2] == '0'))

    {
        Sym.amplitude = 1;
    } 
    else if ((TBits.bitstring[0] == '1') && (Bits.bitstring[1] == '1') && (Bits.bitstring[2] == '1'))

    {
        Sym.amplitude = 2;
    } 
    else if ((TBits.bitstring[0] == '1') && (TBits.bitstring[1] == '0') && (TBits.bitstring[2] == '1'))

    {
        Sym.amplitude = 3;
    } 
    else if ((TBits.bitstring[0] == '1') && (TBits.bitstring[1] == '0') && (TBits.bitstring[2] == '0'))

    {
        Sym.amplitude = 4;
    } 
}

void ImaginaryModulation(BitStream &TBits, Symbol &Sym)
{ 
    if ((TBits.bitstring[3] == '0') && (TBits.bitstring[4] == '0') && (TBits.bitstring[5] == '0'))

    {
        Sym.phase = -4;
    }

    else if ((TBits.bitstring[3] == '0') && (TBits.bitstring[4] == '0') && (TBits.bitstring[5] == '1'))

    {
        Sym.phase = -3;
    } 
    else if ((TBits.bitstring[3] == '0') && (TBits.bitstring[4] == '1') && (TBits.bitstring[5] == '1'))

    {
        Sym.phase = -2;
    } 
    else if ((TBits.bitstring[3] == '0') && (TBits.bitstring[4] == '1') && (TBits.bitstring[5] == '0'))

    {
        Sym.phase = -1;
    } 
    else if ((TBits.bitstring[3] == '1') && (Bits.bitstring[4] == '1') && (TBits.bitstring[5] == '0'))

    {
        Sym.phase = 1;
    } 
    else if ((TBits.bitstring[3] == '1') && (TBits.bitstring[4] == '1') && (TBits.bitstring[5] == '1'))

    {
        Sym.phase = 2;
    } 
    else if ((TBits.bitstring[3] == '1') && (TBits.bitstring[4] == '0') && (TBits.bitstring[5] == '1'))

    {
        Sym.phase = 3;
    } 
    else if ((TBits.bitstring[3] == '1') && (TBits.bitstring[4] == '0') && (TBits.bitstring[5] == '0'))

    {
        Sym.phase = 4;
    } 
}

void RealDeModulation(Symbol& Sym, BitStream& RBits)
{
    if (Sym.amplitude==-4)

    {
        Rbits[0]='0' && Rbits[1]='0' &&Rbits[2]= '0' ;
    }

    else if (Sym.amplitude==-3)

    {
        Rbits[0]='0' && Rbits[1]='0' &&Rbits[2]= '1'
    } 
    else if (Sym.amplitude==-2)

    {
        Rbits[0]='0' && Rbits[1]='1' &&Rbits[2]= '1' ;
    } 
    else if (Sym.amplitude==-1)

    {
        Rbits[0]='0' && Rbits[1]='1' &&Rbits[2]= '0' ;
    } 
    else if (Sym.amplitude==1)

    {
        Rbits[0]='1' && Rbits[1]='1' &&Rbits[2]= '0' ;
    } 
    else if (Sym.amplitude==2)

    {
        Rbits[0]='1' && Rbits[1]='1' &&Rbits[2]= '1' ;
    } 
    else if (Sym.amplitude==3)

    {
        Rbits[0]='1' && Rbits[1]='0' &&Rbits[2]= '1' ;
    } 
    else if (Sym.amplitude==4)

    {
        Rbits[0]='1' && Rbits[1]='0' &&Rbits[2]= '0' ;
    } 
}

void ImaginaryDeModulation(Symbol& Sym, BitStream& RBits)
{
    if (Sym.phase==-4)

    {
        Rbits[3]='0' && Rbits[4]='0' &&Rbits[5]= '0' ;
    }

    else if (Sym.phase==-3)

    {
        Rbits[3]='0' && Rbits[4]='0' &&Rbits[5]= '1'
    } 
    else if (Sym.phase==-2)

    {
        Rbits[3]='0' && Rbits[4]='1' &&Rbits[5]= '1' ;
    } 
    else if (Sym.phase==-1)

    {
        Rbits[3]='0' && Rbits[4]='1' &&Rbits[5]= '0' ;
    } 
    else if (Sym.phase==1)

    {
        Rbits[3]='1' && Rbits[4]='1' &&Rbits[5]= '0' ;
    } 
    else if (Sym.phase==2)

    {
        Rbits[3]='1' && Rbits[4]='1' &&Rbits[5]= '1' ;
    } 
    else if (Sym.phase==3)

    {
        Rbits[3]='1' && Rbits[4]='0' &&Rbits[5]= '1' ;
    } 
    else if (Sym.phase==4)

    {
        Rbits[3]='1' && Rbits[4]='0' &&Rbits[5]= '0' ;
    }
}

void Decoding()
{

}

void NoiseAddition()
{

}


int main() { 
    vector<string> inputs(10); // Vector to store 10 inputs 
    cout << "Enter your inputs:" << endl; 

    // Loop to get 10 inputs from the user
    for (int i = 0; i < 10; i++) { 
        string temp; 
        while (true) {
            cout << "Input " << i + 1 << " : ";
            getline(cin, temp); 

            if (temp.length() == 6) { 
                inputs[i] = temp; 
                break;
            } else {
                cout << "Error: input must be exactly 6 characters long, try again." << endl;
            }
        }
    }

    // Now print all the 10 inputs after they have been entered
    cout << "You entered the following inputs:" << endl; 
    for (int i = 0; i < 10; i++) {
        cout << inputs[i] << endl; // Access inputs using index 
    }
    
    return 0;
}
