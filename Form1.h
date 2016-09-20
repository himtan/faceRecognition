#pragma once

namespace facerecognition {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for Form1
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}
		
	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^  button1;
	protected: 
	private: System::Windows::Forms::Button^  button2;
	private: System::Windows::Forms::Button^  button3;
	private: System::Windows::Forms::TextBox^  textBox1;
	private: System::Windows::Forms::TextBox^  textBox2;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::RichTextBox^  richTextBox1;
	private: System::Windows::Forms::RichTextBox^  richTextBox2;
	private: System::Windows::Forms::Button^  button4;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::Label^  label5;



	protected: 



	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->textBox2 = (gcnew System::Windows::Forms::TextBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->richTextBox1 = (gcnew System::Windows::Forms::RichTextBox());
			this->richTextBox2 = (gcnew System::Windows::Forms::RichTextBox());
			this->button4 = (gcnew System::Windows::Forms::Button());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(21, 25);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(75, 23);
			this->button1->TabIndex = 0;
			this->button1->Text = L"TRAIN";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &Form1::button1_Click);
			// 
			// button2
			// 
			this->button2->Location = System::Drawing::Point(186, 25);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(75, 23);
			this->button2->TabIndex = 1;
			this->button2->Text = L"TEST";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &Form1::button2_Click);
			// 
			// button3
			// 
			this->button3->Location = System::Drawing::Point(126, 368);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(75, 23);
			this->button3->TabIndex = 2;
			this->button3->Text = L"EXIT";
			this->button3->UseVisualStyleBackColor = true;
			this->button3->Click += gcnew System::EventHandler(this, &Form1::button3_Click);
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(12, 64);
			this->textBox1->Name = L"textBox1";
			this->textBox1->Size = System::Drawing::Size(100, 20);
			this->textBox1->TabIndex = 3;
			// 
			// textBox2
			// 
			this->textBox2->Location = System::Drawing::Point(12, 123);
			this->textBox2->Name = L"textBox2";
			this->textBox2->Size = System::Drawing::Size(100, 20);
			this->textBox2->TabIndex = 4;
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(12, 87);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(66, 13);
			this->label1->TabIndex = 5;
			this->label1->Text = L"no of person";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(12, 146);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(93, 13);
			this->label2->TabIndex = 6;
			this->label2->Text = L"images per person";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(342, 25);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(175, 13);
			this->label3->TabIndex = 7;
			this->label3->Text = L"FACE RECOGNITION USING ANN";
			// 
			// richTextBox1
			// 
			this->richTextBox1->BackColor = System::Drawing::SystemColors::ActiveCaptionText;
			this->richTextBox1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->richTextBox1->ForeColor = System::Drawing::Color::LimeGreen;
			this->richTextBox1->Location = System::Drawing::Point(345, 64);
			this->richTextBox1->Name = L"richTextBox1";
			this->richTextBox1->Size = System::Drawing::Size(429, 261);
			this->richTextBox1->TabIndex = 8;
			this->richTextBox1->Text = L"";
			// 
			// richTextBox2
			// 
			this->richTextBox2->BackColor = System::Drawing::SystemColors::ActiveCaptionText;
			this->richTextBox2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->richTextBox2->ForeColor = System::Drawing::Color::LimeGreen;
			this->richTextBox2->Location = System::Drawing::Point(345, 352);
			this->richTextBox2->Name = L"richTextBox2";
			this->richTextBox2->Size = System::Drawing::Size(291, 39);
			this->richTextBox2->TabIndex = 9;
			this->richTextBox2->Text = L"";
			// 
			// button4
			// 
			this->button4->Location = System::Drawing::Point(696, 352);
			this->button4->Name = L"button4";
			this->button4->Size = System::Drawing::Size(75, 23);
			this->button4->TabIndex = 10;
			this->button4->Text = L"enter";
			this->button4->UseVisualStyleBackColor = true;
			this->button4->Click += gcnew System::EventHandler(this, &Form1::button4_Click);
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(351, 48);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(43, 13);
			this->label4->TabIndex = 11;
			this->label4->Text = L"output>";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(351, 336);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(36, 13);
			this->label5->TabIndex = 12;
			this->label5->Text = L"input>";
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::LightSteelBlue;
			this->ClientSize = System::Drawing::Size(821, 414);
			this->Controls->Add(this->label5);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->button4);
			this->Controls->Add(this->richTextBox2);
			this->Controls->Add(this->richTextBox1);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->textBox2);
			this->Controls->Add(this->textBox1);
			this->Controls->Add(this->button3);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->button1);
			this->Name = L"Form1";
			this->Text = L"Face Recognition";
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
private:
	int noOfPerson_h;
	int imagesPerPerson_h;
	int enterButtonCount;
	static int testCount=0;
	static int trainCounter=0;
	static int colourCounter=0;
	private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
				if(button1->Text=="REFRESH")
				{
				WinExec("facerecognition.exe", SW_SHOW);
				 exit(0);
				}
				consoleOutput("\ntraining...");
				System::String^ s1=textBox1->Text;
				System::String^ s2=textBox2->Text;
				clearOutput();
				try
				{
				noOfPerson_h= int::Parse(s1);
				imagesPerPerson_h= int::Parse(s2);
				}
				catch(...)
				{
				consoleOutput("\nwrong format of numbers");
				return;
				}
				if(noOfPerson_h<2||imagesPerPerson_h<5)
				{
				consoleOutput("\nusage:-\nno of person should be greater than 2\nimages per person should be greater than 5");
				}
			 enterButtonCount=0;
			 consoleOutput("\nenter the name of person no ");
		     consoleOutput(enterButtonCount);
		     consoleOutput(" ");
			 richTextBox2->Text="";
			 trainCounter++;
			 
			 }
	private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e) {
				 if(button2->Text=="REFRESH")
				 {
				 WinExec("facerecognition.exe", SW_SHOW);
				 exit(0);
				 }
				 consoleOutput("\ntesting...");
				 pca::pcamain(0,0,1);
				 testCount++;
				 button2->Text="REFRESH";
			 }
	private: System::Void button3_Click(System::Object^  sender, System::EventArgs^  e) { 
				 Application::Exit();
				 std::exit(0);
			 }
	public: void consoleOutput(System::String^ c)
		{
		 System::String^ s=richTextBox1->Text;
		 s+=c;
		 richTextBox1->Text=s;
		 richTextBox1->SelectionStart = (richTextBox1->Text)->Length;
		 richTextBox1->ScrollToCaret();
		}
	public: void consoleOutput(double d)
		{
			System::String^c=(gcnew System::Double(d))->ToString();
		 System::String^ s=richTextBox1->Text;
		 s+=c;
		 richTextBox1->Text=s;
		 richTextBox1->SelectionStart = (richTextBox1->Text)->Length;
		 richTextBox1->ScrollToCaret();
		}
		public: void clearOutput()
		{
		richTextBox1->Text="";
		}
		public: void clearInput()
		{
		richTextBox2->Text="";
		}	
				public:void changeColour()
					   {
						   if(colourCounter%7==0)
							   this->BackColor=System::Drawing::Color::LightYellow;
						   else if(colourCounter%7==1)
							   this->BackColor=System::Drawing::Color::RosyBrown;
						   else if(colourCounter%7==2)
							   this->BackColor=System::Drawing::Color::Salmon;
						   else if(colourCounter%7==3)
							   this->BackColor=System::Drawing::Color::YellowGreen;
						   else if(colourCounter%7==4)
							   this->BackColor=System::Drawing::Color::SkyBlue;
						   else if(colourCounter%7==5)
							   this->BackColor=System::Drawing::Color::MediumSlateBlue;
						   else if(colourCounter%7==6)
							   this->BackColor=System::Drawing::Color::LightSteelBlue;
						   colourCounter++;
					   }
	private: System::Void button4_Click(System::Object^  sender, System::EventArgs^  e) {
			 static std::ofstream ofName;
			 if(enterButtonCount==0)
				 ofName.open("names.dat");
			 enterButtonCount++;
			 char* name = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(richTextBox2->Text).ToPointer();
			 ofName<<name;
			 ofName<<'#';
			 clearInput();
			if(enterButtonCount==noOfPerson_h)
			 {		
				 ofName.close();
				 pca::pcamain(noOfPerson_h,imagesPerPerson_h,0);
				 button1->Text="REFRESH";
				 button2->Text="REFRESH";

			}
			else
			{
			consoleOutput("\nenter the name of person no ");
		    consoleOutput(enterButtonCount);
		    consoleOutput(" ");
			}

			 }
};
}

