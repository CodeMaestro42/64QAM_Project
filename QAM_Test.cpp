#include <iostream>
#include <cmath> 
#include <random>
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std; 

class InputBitStream {

public: 
    string inputstream; 
    InputBitStream() : inputstream("000");
};

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

void ErrorDetect()
{}

void NoiseAddition()
{

}

int main() {
    // Seed the random number generator
    srand(time(0)); 

    int length;
    cout << "Enter the length of each string: ";
    cin >> length;

    // Vector to store the random strings
    vector<string> randomStrings;

    for (int j = 0; j < 10; j++) { // Loop to generate 10 strings
        string randomString = "";

        for (int i = 0; i < length; i++) {
            // Generate a random number, either 0 or 1
            int randomBit = rand() % 2; 

            // Append the random bit to the string
            randomString += to_string(randomBit); 
        }

        // Store the random string in the vector
        randomStrings.push_back(randomString);
    }

    // Output all stored random strings
    for (int i = 0; i < randomStrings.size(); i++) {
        cout << "Random string " << i + 1 << ": " << randomStrings[i] << endl;
    }

    return 0;
}
