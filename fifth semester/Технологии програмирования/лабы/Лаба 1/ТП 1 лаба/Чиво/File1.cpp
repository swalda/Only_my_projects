/////////////////////////////////////////////////////////////////////////////////////////
//Деление с остатком полиномов от трех переменных.
/////////////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <cmath>
#include <set>
#include <string>
#include <utility>
/////////////////////////////////////////////////////////////////////////////////////////
typedef std::string  T_str;
typedef double       T_coef;
typedef int          T_degree;
/////////////////////////////////////////////////////////////////////////////////////////
struct  T_monom
{
    T_coef      coef_;
    T_degree    X_degree_;
    T_degree    Y_degree_;
    T_degree    Z_degree_;
    //-----------------------------------------------------------------------------------
    T_monom
        (
            T_coef      coef        = 0,
            T_degree    X_degree    = 0,
            T_degree    Y_degree    = 0,
            T_degree    Z_degree    = 0
        )
        :
        coef_       (coef),
        X_degree_   (X_degree),
        Y_degree_   (Y_degree),
        Z_degree_   (Z_degree)
    {}
    //-----------------------------------------------------------------------------------
    bool        divides     (const T_monom&  monom)     const
    {
        return      X_degree_ <= monom.X_degree_
                &&  Y_degree_ <= monom.Y_degree_
                &&  Z_degree_ <= monom.Z_degree_;
    }
    //-----------------------------------------------------------------------------------
    T_monom     operator*   (const T_monom&  monom)     const
    {
        T_monom  res
            (
                coef_       *   monom.coef_,
                X_degree_   +   monom.X_degree_,
                Y_degree_   +   monom.Y_degree_,
                Z_degree_   +   monom.Z_degree_
            );
        return  res;
    }
    //-----------------------------------------------------------------------------------
    T_monom     operator/   (const T_monom&  monom)     const
    {
        T_monom  res
            (
                coef_       /   monom.coef_,
                X_degree_   -   monom.X_degree_,
                Y_degree_   -   monom.Y_degree_,
                Z_degree_   -   monom.Z_degree_
            );
        return  res;
    }
    //-----------------------------------------------------------------------------------
    T_monom     operator-   ()                          const
    {
        T_monom  res = *this;
        res.coef_ *= -1;
        return  res;
    }
    //-----------------------------------------------------------------------------------
    bool        operator==  (const T_monom&  monom)     const
    {
        return      coef_       ==  monom.coef_
                &&  X_degree_   ==  monom.X_degree_
                &&  Y_degree_   ==  monom.Y_degree_
                &&  Z_degree_   ==  monom.Z_degree_;
    }
    //-----------------------------------------------------------------------------------
    bool        operator!=  (const T_monom&  monom)     const
    {
        return  !(*this == monom);
    }
    //-----------------------------------------------------------------------------------
    bool        operator<   (const T_monom&  monom)     const
    {
        //Сортируем так, чтобы в начале стояли старшие члены.
        return      std::make_pair
                        (
                            X_degree_,
                            std::make_pair
                                (
                                    Y_degree_,
                                    Z_degree_
                                )
                        )

                >   std::make_pair
                        (
                            monom.X_degree_,
                            std::make_pair
                                (
                                    monom.Y_degree_,
                                    monom.Z_degree_
                                )
                        );
    }
};
/////////////////////////////////////////////////////////////////////////////////////////
typedef std::multiset<T_monom>  T_polynom_XYZ;
/////////////////////////////////////////////////////////////////////////////////////////
void  input_monom_var_degree
    (
        const T_str&    prompt,
        T_degree&       degree
    )
{
    do
    {
        std::cout   <<  prompt;
        std::cin    >>  degree;
    }while(degree < 0);
}
/////////////////////////////////////////////////////////////////////////////////////////
T_polynom_XYZ  input_polynom_XYZ(const T_str&  prompt)
{
    std::cout << prompt
              << std::endl;

    int  polynom_size = 0;
    do
    {
        std::cout << "Введите количество одночленов в многочлене от переменных X, Y, Z: ";
        std::cin >> polynom_size;
    }while(polynom_size <= 0);

    std::cout << std::endl
              << "Введите "
              << polynom_size
              << " одночленов от переменных X, Y, Z."
              << std::endl;

    T_polynom_XYZ  polynom;
    for(int  i = 0; i < polynom_size; ++i)
    {
        T_monom  cur_monom;
        std::cout << std::endl
                  << "одночлен #"
                  << i + 1
                  << ":"
                  << std::endl;

        std::cout<< "\tкоэффициент:\t";
        std::cin >> cur_monom.coef_;

        input_monom_var_degree
            (
                "\tстепень X:\t",
                cur_monom.X_degree_
            );

        input_monom_var_degree
            (
                "\tстепень Y:\t",
                cur_monom.Y_degree_
            );

        input_monom_var_degree
            (
                "\tстепень Z:\t",
                cur_monom.Z_degree_
            );

        polynom.insert(cur_monom);
    }
    std::cout << std::endl
              << std::endl;
    return  polynom;
}
/////////////////////////////////////////////////////////////////////////////////////////
void  trim
    (
        T_polynom_XYZ&  polynom,
        T_coef          my_epsilon
    )
{
    if( !polynom.empty() )
    {
        T_polynom_XYZ  new_polynom;
        //Обходим полином, и все одночлены копируем в new_polynom, при этом
        //одночлены с одинаковыми степенями переменных объединяем в один,
        //суммируя их коэффициенты.
        for(T_polynom_XYZ::const_iterator  elem_it = polynom.begin();
            elem_it != polynom.end(); )
        {
            T_polynom_XYZ::value_type  cur_monom = *elem_it;
            //Заводим одночлен для суммирования коэффициентов.
            T_polynom_XYZ::value_type  new_monom = cur_monom;
            ++elem_it;
            //Обходим все степенные дубликаты текущего одночлена.
            for(; elem_it != polynom.upper_bound(cur_monom); ++elem_it)
            {
                new_monom.coef_ += elem_it->coef_;
            }

            //Если коэффициент нового одночлена не очень близок к нулю,
            //то копируем этот одночлен в новый полином.
            if(abs(new_monom.coef_)  >   my_epsilon)
            {
                new_polynom.insert(new_monom);
            }
        }
        std::swap(polynom, new_polynom);
    }//if( !polynom.empty() )

    //Если полином опустел после суммирования коэффициентов, или был пуст изначально, то.
    if( polynom.empty() )
    {
        polynom.insert( T_monom() );
    }
}
/////////////////////////////////////////////////////////////////////////////////////////
void  divide
    (
        T_polynom_XYZ   L,
        T_polynom_XYZ   R,
        T_polynom_XYZ&  quotient,
        T_polynom_XYZ&  remainder,
        T_coef          my_epsilon
    )
{
    trim(L, my_epsilon);
    trim(R, my_epsilon);

    quotient.clear();

    bool  bool_res = true;
    while(
            (
                    L.size()    >   1
                ||  *L.begin()  !=  T_monom()
            )
            &&  bool_res
         )
    {
        bool_res = false;
        T_polynom_XYZ::const_iterator  elem_for_div_it;
        //Ищем самый старший член в L, который делится на старший член R.
        for(T_polynom_XYZ::const_iterator  elem_it = L.begin();
            elem_it != L.end(); ++elem_it)
        {
            if( R.begin()->divides(*elem_it) )
            {
                bool_res            = true;
                elem_for_div_it     = elem_it;
                break;
            }
        }

        if(bool_res)
        {
            //Вычисляем множитель.
            T_monom  factor = *elem_for_div_it / *R.begin();
            //Добавляем его в результат.
            quotient.insert(factor);

            //Вычитаем из L делитель R, умноженный на factor.
            for(T_polynom_XYZ::const_iterator  R_elem_it = R.begin();
                R_elem_it != R.end(); ++R_elem_it)
            {
                L.insert(-*R_elem_it * factor);
            }
            trim(L, my_epsilon);
        }
    }

    remainder = L;
    trim(quotient, my_epsilon);
}
/////////////////////////////////////////////////////////////////////////////////////////
void  print_polynom
    (
        const T_str&            prompt,
        const T_polynom_XYZ&    polynom
    )
{
    std::cout << std::endl
              << prompt
              << std::endl;

    for(T_polynom_XYZ::const_iterator  elem_it = polynom.begin();
        elem_it != polynom.end(); ++elem_it)
    {
        std::cout << std::showpos
                  << elem_it->coef_
                  << "*x^"
                  << std::noshowpos
                  << elem_it->X_degree_
                  << "*y^"
                  << elem_it->Y_degree_
                  << "*z^"
                  << elem_it->Z_degree_
                  << std::endl;
    }
    std::cout << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////
int main()
{
    std::locale::global(std::locale(""));
    const T_coef    MY_EPSILON  =   std::numeric_limits<T_coef>::epsilon() * 10;
    T_polynom_XYZ   L           =   input_polynom_XYZ("ВВОД ПОЛИНОМА-ДЕЛИМОГО.");
    T_polynom_XYZ   R           =   input_polynom_XYZ("ВВОД ПОЛИНОМА-ДЕЛИТЕЛЯ.");

    T_polynom_XYZ   quotient;
    T_polynom_XYZ   remainder;

    divide
        (
            L,
            R,
            quotient,
            remainder,
            MY_EPSILON
        );

    print_polynom
        (
            "Полином-частное:",
            quotient
        );

    print_polynom
        (
            "Полином-остаток:",
            remainder
        );
}
