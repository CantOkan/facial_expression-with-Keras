int pin=13;
int pin2=12;
char userInput;

// the setup routine runs once when you press reset:
void setup() {
  
  pinMode(pin,OUTPUT);
  pinMode(pin2,OUTPUT);

  Serial.begin(9600);
}

// the loop routine runs over and over again forever:
void loop() {

if(Serial.available()>0)
  {
    userInput=Serial.read();

    if(userInput =='g' || userInput =='N')
    {
             
      digitalWrite(pin2,LOW);
      digitalWrite(pin,HIGH);

    }
    else if(userInput=='H')
    {
       digitalWrite(pin,LOW);
       digitalWrite(pin2,HIGH);


    }

    
  }

}
