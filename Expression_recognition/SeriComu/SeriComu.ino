int analogpin=A3;
int data=0;
char userInput;

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

// the loop routine runs over and over again forever:
void loop() {

if(Serial.available()>0)
  {
    userInput=Serial.read();

    if(userInput =='sad')
    {
      int sensorValue = analogRead(A0);
      data=analogRead(analogpin);
      Serial.println(data);
      
      
    }
  }



}
