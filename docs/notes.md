
1. At first, I was thinking of making a personal tool, but after bouncing the problem around with my friends, they were all in support and inspired me to make this an actual well structured and functional natural language processing document processer. 


I do not want to create a specialized IDE to write text, just for text, because users do just fine using VSCode or Pycharm just for writing text. An ide is not necessary, but not avoidable. There still has to be a structured way to organise extremely large and detailed text documents without the need for IDEs. 

I was now thinking of converting this into a rust based project, to ensure speed and performance. The original system based on highlighting text was stupid.  Imagine the following : 



I have a folder with a bunch of text files. I want to process that text/information and use it in a complex downstream pipeline ? well, in the same way one writes a python script with function definations and detailed business logic that leads to specific operations ? well, that is what turbulance is, a quasi programming language/scripting language ...that is, python for text. One should be able to write steps in code on the intended text operations.

Here is an example 
`funxn (start, stop, e ):
      within start and stop:
       given e == "semantic error":
                  < semantic type a blah
       processed_string = new 
       return (new)
`
Now, if I ran this function over all my text, I expect, only the above mentioned cases to be operated on. That is the point. At first, I was considering them to be keyboard shortcuts, but realised that, a whole new scripting language would work. 

Text, unlike numbers, has blurry boundaries unless if a user made a boundary that constituted a "unit", and could process it in a way any unit should be processed in a complete axiom. 