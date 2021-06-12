

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;


namespace ConsoleApp1
{
   class Node
    {   
        public Node Prev;
        public Node Next;
        public float koef;
        public int PowX;
        public int PoweX;
        public Node(float k = 0, int X = 0, int eX = 0, Node pr = null, Node nx = null)
        {
            Prev = pr;
            Next = nx;
            koef = k;
            PowX = X;
            PoweX = eX;
        } 
        public Node Copy()
        {
            Node n = this;
            Node res = new Node();
            Node q = res;
            Node t = n;

            q.koef = t.koef;
            q.PowX = t.PowX;
            q.PoweX = t.PoweX;
            q.Next = t.Next;
            q.Prev = t.Prev;
            t = t.Next;
            while(t != null)
            {
                q = new Node();
                q.koef = t.koef;
                q.PowX = t.PowX;
                q.PoweX = t.PoweX;
                q.Next = t.Next;
                q.Prev = t.Prev;
                t = t.Next;
            }
            return res;
        }
        public static Node Copy(Node n)
        {
            Node res = new Node();
            Node q = res;
            Node t = n;

            if (t != null)
            {
                q.koef = t.koef;
                q.PowX = t.PowX;
                q.PoweX = t.PoweX;
                q.Next = null;
                q.Prev = null;
                t = t.Next;
            }
            while (t != null)
            {
                Node last = q;
                q = new Node();
                q.koef = t.koef;
                q.PowX = t.PowX;
                q.PoweX = t.PoweX;
                q.Next = null;
                q.Prev = last;
                last.Next = q;
                t = t.Next;
            }
            return res;
        }
    }

    class Polynome
    {
        private Node Head;
        public Polynome()
        {
            Head = new Node();
            this.ReFormating();
        }
        public Polynome(string filename)
        {
            Head = new Node();
            Node now = Head;
            FileStream file1 = new FileStream(filename, FileMode.Open, FileAccess.Read); //открывает файл на чтение
            StreamReader reader = new StreamReader(file1);

            int n = Convert.ToInt32(reader.ReadLine());

            Head.koef = Convert.ToInt32(reader.ReadLine());
            Head.PowX = Convert.ToInt32(reader.ReadLine());
            Head.PoweX = Convert.ToInt32(reader.ReadLine());
            n -= 1;
            
            while (n != 0)
            {
                Node temp = new Node();
                temp.koef = Convert.ToInt32(reader.ReadLine());
                temp.PowX = Convert.ToInt32(reader.ReadLine());
                temp.PoweX = Convert.ToInt32(reader.ReadLine());
                temp.Next = null;
                temp.Prev = now;
                now.Next = temp;
                now = temp;
                n -= 1;
            }

            this.ReFormating();
        }
        public Polynome(Node nod)
        {
            Node t = Node.Copy(nod);
            Head = t;
        }
        public void print()
        {
            if (Head != null)
            {
                Node t = Head;
                Console.Write("({0:F} * x^{1:D} * (e^x)^{2:D}) ", t.koef, t.PowX, t.PoweX);
                t = t.Next;
                while (t != null)
                {
                    Console.Write("+ ({0:F} * x^{1:D} * (e^x)^{2:D}) ", t.koef, t.PowX, t.PoweX);
                    t = t.Next;
                }
                Console.WriteLine("");
            }
            else
            {
                Console.WriteLine("Нулевой многочлен");
            }
        }
        private void ReFormating()
        {
            Node t, j;
            t = Head;
            while (t != null)
            {
                j = t.Next;
                while (j != null)
                {
                    if (j.PowX == t.PowX && j.PoweX == t.PoweX)
                    {
                        j.Prev.Next = j.Next;
                        if (j.Next != null)
                            j.Next.Prev = j.Prev;
                        t.koef = t.koef + j.koef;
                    }
                    j = j.Next;
                }
                t = t.Next;
            }

            t = Head;
            while (t != null)
            {
                if (t.koef == 0)
                {
                    if (t.Prev == null)
                    {
                        Head = Head.Next;
                        if (Head != null)
                        {
                            Head.Prev = null;
                        }
                    }
                    else if (t.Next == null)
                    {
                        t.Prev.Next = null;
                    }
                    else
                    {
                        t.Next.Prev = t.Prev;
                        t.Prev.Next = t.Next;
                    }
                }
                t = t.Next;
            }

            t = Head;
            while (t != null)
            {
                j = t.Next;
                while (j != null)
                {
                    if (j.PowX > t.PowX)
                    {
                        int tmp = 0;
                        float tmpf = 0;

                        tmp = t.PowX;
                        t.PowX = j.PowX;
                        j.PowX = tmp;

                        tmp = t.PoweX;
                        t.PoweX = j.PoweX;
                        j.PoweX = tmp;

                        tmpf = t.koef;
                        t.koef = j.koef;
                        j.koef = tmpf;
                    }
                    j = j.Next;
                }
                t = t.Next;
            }
        }
        public static Polynome operator +(Polynome pol1, Polynome pol2)
        {
            Polynome res = new Polynome();
            Node t1 = Node.Copy(pol1.Head);
            Node t2 = Node.Copy(pol2.Head);
            Node Lastoft1 = t1;
            while (Lastoft1.Next != null)
                Lastoft1 = Lastoft1.Next;
            Lastoft1.Next = t2;
            t2.Prev = Lastoft1;
            res.Head = t1;
            res.ReFormating();
            return res;
        }
        public static Polynome operator *(Polynome pol1, float k)
        {
            Polynome res = new Polynome();
            Node t1 = Node.Copy(pol1.Head);
            Node q = t1;
            while(q != null)
            {
                q.koef = q.koef * k;
                q = q.Next;
            }
            res.Head = t1;
            res.ReFormating();
            return res;
        }
        public static Polynome operator *(Polynome pol1, Polynome pol2)
        {
            Node res = new Node();
            Node t1 = Node.Copy(pol1.Head);
            Node t2 = Node.Copy(pol2.Head);
            Node q = res;
            while (t1 != null)
            {
                while(t2 != null)
                {
                    Node tmp = new Node();
                    tmp.koef = t1.koef * t2.koef;
                    tmp.PowX = t1.PowX + t2.PowX;
                    tmp.PoweX = t1.PoweX + t2.PoweX;
                    tmp.Next = null;
                    tmp.Prev = q;
                    q.Next = tmp;
                    q = q.Next;
                    t2 = t2.Next;
                }
                t2 = Node.Copy(pol2.Head);
                t1 = t1.Next;
            }

            Polynome resu = new Polynome(res);
            resu.ReFormating();
            return resu;
        }
        public static Polynome operator *(float k, Polynome pol1)
        {
            Polynome res = new Polynome();
            Node t1 = Node.Copy(pol1.Head);
            Node q = t1;
            while (q != null)
            {
                q.koef = q.koef * k;
                q = q.Next;
            }
            res.Head = t1;
            res.ReFormating();
            return res;
        }
        public static Polynome operator -(Polynome pol1, Polynome pol2)
        {
            Polynome res = new Polynome();
            Node t1 = Node.Copy(pol1.Head);
            Node t2 = Node.Copy(pol2.Head);
            Node Lastoft1 = t1;
            while (Lastoft1.Next != null)
                Lastoft1 = Lastoft1.Next;
            Node q = t2;
            while (q != null)
            {
                q.koef = -q.koef;
                q = q.Next;
            }
            Lastoft1.Next = t2;
            t2.Prev = Lastoft1;
            res.Head = t1;
            res.ReFormating();
            return res;
        }
        public static Polynome operator /(Polynome pol1, Polynome pol2)
        {
            Node resul = new Node();
            Node q = resul;
            Polynome Pol1copy = new Polynome(Node.Copy(pol1.Head));

            
            while(Pol1copy.Head != null && (Pol1copy.Head.PowX >= pol2.Head.PowX && Pol1copy.Head.PoweX >= pol2.Head.PoweX))
            {
                if (Pol1copy.Head.PowX >= pol2.Head.PowX && Pol1copy.Head.PoweX >= pol2.Head.PoweX)
                {
                    Node now = new Node();

                    now.koef = Pol1copy.Head.koef / pol2.Head.koef;
                    now.PowX = Pol1copy.Head.PowX - pol2.Head.PowX;
                    now.PoweX = Pol1copy.Head.PoweX - pol2.Head.PoweX;

                    Polynome now1 = new Polynome(now);
                    Pol1copy = Pol1copy - (pol2 * now1);

                    q.Next = now;
                    now.Next = null;
                    now.Prev = q;
                    q = q.Next;
                }
            }

            Polynome res = new Polynome(resul);
            res.ReFormating();
            return res;
        }
        public static Polynome operator %(Polynome pol1, Polynome pol2)
        {
            Node resul = new Node();
            Node q = resul;
            Polynome Pol1copy = new Polynome(Node.Copy(pol1.Head));


            while (Pol1copy.Head != null && (Pol1copy.Head.PowX >= pol2.Head.PowX && Pol1copy.Head.PoweX >= pol2.Head.PoweX))
            {
                if (Pol1copy.Head.PowX >= pol2.Head.PowX && Pol1copy.Head.PoweX >= pol2.Head.PoweX)
                {
                    Node now = new Node();

                    now.koef = Pol1copy.Head.koef / pol2.Head.koef;
                    now.PowX = Pol1copy.Head.PowX - pol2.Head.PowX;
                    now.PoweX = Pol1copy.Head.PoweX - pol2.Head.PoweX;

                    Polynome now1 = new Polynome(now);
                    Pol1copy = Pol1copy - (pol2 * now1);

                    q.Next = now;
                    now.Next = null;
                    now.Prev = q;
                    q = q.Next;
                }
            }

            Polynome res = new Polynome(Pol1copy.Head);
            res.ReFormating();
            return res;
        }
    }

    class Program
    {
        static void Main()
        {
            Polynome MyPol1 = new Polynome("Data1.txt");
            Polynome MyPol2 = new Polynome("Data2.txt");

            Console.WriteLine("Первый многочлен: ");
            MyPol1.print();
            Console.WriteLine("Второй многочлен: ");
            MyPol2.print();

            Polynome MyPol3 = MyPol1 + MyPol2;
            Console.WriteLine("Сумма многочленов: ");
            MyPol3.print();

            Polynome MyPol4 = MyPol1 - MyPol2;
            Console.WriteLine("Разность многочленов: ");
            MyPol4.print();

            Console.WriteLine("Умножаем первый многочлен на число, введите его: ");
            float a = Convert.ToSingle(Console.ReadLine());
            Polynome MyPol5 = a * MyPol1;
            MyPol5.print();

            Polynome MyPol6 = MyPol1 / MyPol2;
            Console.WriteLine("Отношение первого и второго многочленов без остатка: ");
            MyPol6.print();

            Polynome MyPol7 = MyPol1 % MyPol2;
            Console.WriteLine("Остаток от деления первого многочлена на второй: ");
            MyPol7.print();

            Polynome MyPol8 = MyPol1 * MyPol2;
            Console.WriteLine("Произведение первого и второго многочленов: ");
            MyPol8.print();

            Console.Read();
        }
    }
}
