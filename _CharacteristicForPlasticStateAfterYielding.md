
# Equations

 \begin{equation}\label{eq:1d}
   \left\{ \begin{aligned}
       & \partial _t \rho +\partial_x(\rho u)=0,\\
       & \partial _t (\rho u)+\partial_x(\rho u^2 + p -s_{xx})=0,\\
       &\partial _t (\rho E)+\partial_x\left[(\rho E + p -s_{xx})u\right]=0,\\
       &\partial _t s_{xx}+u\partial_xs_{xx}-\frac{4}{3}\partial_x u=0,\\
& |s_{xx}|\leq \frac{2}{3}Y_{0}, \\
       &Q(x,t = 0) = \left\{\begin{aligned}
           Q_L, \hspace{0.1cm} \text{if} \hspace{0.1cm} x<0, \\
           Q_R, \hspace{0.1cm} \text{if} \hspace{0.1cm} x\ge 0, \\
       \end{aligned}\right.
     \end{aligned}
  \right.
\end{equation}


When the material is yielding,
\begin{equation}
  |s_{xx}| = \frac{2}{3}Y_0,
\end{equation}
the system is turned into a more simple system with only constitutive terms
\begin{equation}
  \partial_t \mathbf{{U}} + \partial_x \mathbf{F}= 0,
\end{equation}
where $\mathbf{U} = (\rho, \rho u, \rho E )$ and $\mathbf{F} = \left(\rho u, \rho u^2 +p -sxx, (\rho E+p-sxx)u\right)$.
$$ p = \rho_0 a_0^2 f_\eta +\rho_0 \Gamma_0 (E-\frac{u^2}{2})$$
In the following we will give the Jaccobi matrix of $\frac{\partial \mathbf{F}}{\partial \mathbf{U}}$.


```julia
using SymPy  # you nedd sympy package and your python also need to  have a sympy.
```


```julia
@vars ρ e u Γ σ p_ρ  sxx μ p
```




    (ρ, e, u, Γ, σ, p_ρ, sxx, μ, p)




```julia
@vars x y z 
@vars a_0 ρ_0 Γ_0 Γ
fη= sympy.Function("fη")
fη = fη(x)
```




\begin{equation*}\operatorname{fη}{\left(x \right)}\end{equation*}



$$ U = (\rho,\rho u, \rho E)=(x,y,z) $$
Turn $\mathbf{F}$ into $(x,y,z)$ space.


```julia
f1 =  ρ*u 
f12 = f1(ρ=>x, u=>y/x)
f2 = ρ*u^2 + p-sxx 
f22 = f2(ρ=>x, u=>y/x, p=> ρ_0*a_0^2*fη(x)+ρ_0*Γ_0*(z/x-y^2/x^2/2))
f3 = (ρ*E+p-sxx)*u
f32 = f3(ρ=>x, u=>y/x, E=>z/x, p=> ρ_0*a_0^2*fη(x)+ρ_0*Γ_0*(z/x-y^2/x^2/2))
```




\begin{equation*}\frac{y \left(a_{0}^{2} ρ_{0} \operatorname{fη}{\left(x \right)} - sxx + z + Γ_{0} ρ_{0} \left(\frac{z}{x} - \frac{y^{2}}{2 x^{2}}\right)\right)}{x}\end{equation*}



# 在(x,y ,z) 空间求解Jaccobi矩阵
# Give the Jaccobi in the (x,y,z) space


```julia
F11, F12, F13 = sympy.diff(f12,x), sympy.diff(f12,y),sympy.diff(f12,z)
```




    (0, 1, 0)




```julia
F21, F22, F23 = simplify(sympy.diff(f22,x)), simplify(sympy.diff(f22,y)),sympy.diff(f22,z)
```




    ((a_0^2*x^3*ρ_0*Derivative(fη(x), x) - x*y^2 - Γ_0*ρ_0*(x*z - y^2))/x^3, y*(2*x - Γ_0*ρ_0)/x^2, Γ_0*ρ_0/x)




```julia
F31, F32, F33 = simplify(sympy.diff(f32,x)), simplify(sympy.diff(f32,y)),simplify(sympy.diff(f32,z))
```




    (y*(-2*x^2*(a_0^2*ρ_0*fη(x) - sxx + z) - Γ_0*ρ_0*(2*x*z - y^2) + 2*ρ_0*(a_0^2*x^3*Derivative(fη(x), x) - Γ_0*(x*z - y^2)))/(2*x^4), a_0^2*ρ_0*fη(x)/x - sxx/x + z/x + z*Γ_0*ρ_0/x^2 - 3*y^2*Γ_0*ρ_0/(2*x^3), y*(x + Γ_0*ρ_0)/x^2)



## 变换回 ( ρ, ρu, ρE)空间
# Change back into the $(\rho, \rho u,\rho E)$ space              


```julia
F11 = F11(x=>ρ, y=>ρ*u, z=>ρ*E)
```




\begin{equation*}0\end{equation*}




```julia
F21 = simplify(F21(x=>ρ, y=>ρ*u, z=>ρ*E))
```




\begin{equation*}\frac{a_{0}^{2} ρ ρ_{0} \frac{d}{d ρ} \operatorname{fη}{\left(ρ \right)} - u^{2} ρ + Γ_{0} ρ_{0} \left(u^{2} - e\right)}{ρ}\end{equation*}



$$ a_0^2f\eta\eta - u^2 + \frac{\Gamma_0 \rho_0}{\rho}(-e + \frac{u^2}{2}) $$


```julia
F22 = simplify(F22(x=>ρ, y=>ρ*u, z=>ρ*E))
```




\begin{equation*}- \frac{u Γ_{0} ρ_{0}}{ρ} + 2 u\end{equation*}



$$u(2-\frac{\Gamma_0\rho_0}{\rho})$$


```julia
F23 = simplify(F23(x=>ρ, y=>ρ*u, z=>ρ*E))
```




\begin{equation*}\frac{Γ_{0} ρ_{0}}{ρ}\end{equation*}




```julia
F31 = simplify(F31(x=>ρ, y=>ρ*u, z=>ρ*E))
F31 = simplify(F31(E=>e+u^2/2, Γ_0*ρ_0=> Γ*ρ))
```




\begin{equation*}a_{0}^{2} u ρ_{0} \frac{d}{d ρ} \operatorname{fη}{\left(ρ \right)} - \frac{a_{0}^{2} u ρ_{0} \operatorname{fη}{\left(ρ \right)}}{ρ} - 2 e u Γ - e u + \frac{sxx u}{ρ} + \frac{u^{3} Γ}{2} - \frac{u^{3}}{2}\end{equation*}



$$ \left[ a_0^2 f\eta \eta  + \frac{\sigma}{\rho} -(\Gamma+1)e +\frac{u^2}{2}(\Gamma -1)\right] u $$


```julia
F32 = simplify(F32(x=>ρ, y=>ρ*u, z=>ρ*E, Γ_0* ρ_0 => Γ*ρ))
F32 = simplify(F32(E=>e+u^2/2))
```




\begin{equation*}\frac{a_{0}^{2} ρ_{0} \operatorname{fη}{\left(ρ \right)}}{ρ} + e Γ + e - \frac{sxx}{ρ} - u^{2} Γ + \frac{u^{2}}{2}\end{equation*}



$$-\frac{\sigma}{\rho} +e -u^2\Gamma +\frac{u^2}{2}$$


```julia
F33 = simplify(F33(x=>ρ, y=>ρ*u, z=>ρ*E, Γ_0* ρ_0 => Γ*ρ))
```




\begin{equation*}u \left(Γ + 1\right)\end{equation*}



## 最终形式


```julia
D =sympy.Matrix([0 1 0 ; 
        -u^2+p_ρ+Γ*(u^2/2-e) u*(2-Γ) Γ ; 
        ((Γ-1)*u^2/2-(Γ+1)*e+σ/ρ+p_ρ)*u  u^2/2-Γ*u^2-σ/ρ+e (1+Γ)*u ])
```




\[\left[ \begin{array}{rrr}0&1&0\\p_{ρ} - u^{2} + Γ \left(- e + \frac{u^{2}}{2}\right)&u \left(2 - Γ\right)&Γ\\u \left(- e \left(Γ + 1\right) + p_{ρ} + \frac{u^{2} \left(Γ - 1\right)}{2} + \frac{σ}{ρ}\right)&e - u^{2} Γ + \frac{u^{2}}{2} - \frac{σ}{ρ}&u \left(Γ + 1\right)\end{array}\right]\]



## 求解特征值


```julia
eigvals(D)
```




\[ \left[ \begin{array}{r}\frac{u ρ + \sqrt{p_{ρ} ρ^{2} - Γ ρ σ}}{ρ}\\u\\\frac{u ρ - \sqrt{p_{ρ} ρ^{2} - Γ ρ σ}}{ρ}\end{array} \right] \]



$$ \lambda_1 =u,\quad \lambda_2 = u+c, \quad \lambda_3 = u-c$$ 
where 
$$ c= \sqrt{\frac{\partial p}{\partial \rho}-\frac{\Gamma \sigma}{\rho}}$$

## 求解特征矩阵


```julia
A = eigvecs(D)

```




    3-element Array{Array{Sym,1},1}:
     [Γ/(-p_ρ + u^2 - u*(u*(2 - Γ) - u) - Γ*(-e + u^2/2)), u*Γ/(-p_ρ + u^2 - u*(u*(2 - Γ) - u) - Γ*(-e + u^2/2)), 1]                                                                                                                                                        
     [Γ/(-p_ρ + u^2 - Γ*(-e + u^2/2) - (u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))*(u*(2 - Γ) - (u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/ρ)/ρ), Γ*(u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/(ρ*(-p_ρ + u^2 - Γ*(-e + u^2/2) - (u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))*(u*(2 - Γ) - (u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/ρ)/ρ)), 1]
     [Γ/(-p_ρ + u^2 - Γ*(-e + u^2/2) - (u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))*(u*(2 - Γ) - (u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/ρ)/ρ), Γ*(u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/(ρ*(-p_ρ + u^2 - Γ*(-e + u^2/2) - (u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))*(u*(2 - Γ) - (u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/ρ)/ρ)), 1]




```julia
l11 = simplify(Γ/(-p_ρ + u^2 - u*(u*(2 - Γ) - u) - Γ*(-e + u^2/2)))
```




\begin{equation*}\frac{2 Γ}{2 e Γ - 2 p_{ρ} + u^{2} Γ}\end{equation*}




```julia
l12 = simplify(u*Γ/(-p_ρ + u^2 - u*(u*(2 - Γ) - u) - Γ*(-e + u^2/2)))
```




\begin{equation*}\frac{2 u Γ}{2 e Γ - 2 p_{ρ} + u^{2} Γ}\end{equation*}



$$  l_1 = (1 , u,E -\frac{1}{\Gamma} \frac{\partial p}{\partial \rho})$$

$$  l_1 = (1 , u,E -\frac{c^2}{\Gamma}-\frac{\sigma}{\rho})$$


```julia
l21 = simplify(Γ/(-p_ρ + u^2 - Γ*(-e + u^2/2) - (u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))*(u*(2 - Γ) - (u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/ρ)/ρ))
```




\begin{equation*}\frac{2 ρ}{2 e ρ + u^{2} ρ - 2 u \sqrt{ρ \left(p_{ρ} ρ - Γ σ\right)} - 2 σ}\end{equation*}




```julia
l22 = simplify(Γ*(u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/(ρ*(-p_ρ + u^2 - Γ*(-e + u^2/2) - (u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))*(u*(2 - Γ) - (u*ρ - sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/ρ)/ρ)))
```




\begin{equation*}\frac{2 \left(u ρ - \sqrt{ρ \left(p_{ρ} ρ - Γ σ\right)}\right)}{2 e ρ + u^{2} ρ - 2 u \sqrt{ρ \left(p_{ρ} ρ - Γ σ\right)} - 2 σ}\end{equation*}



$$ l_2 = (1, u- c, E-uc-\frac{\sigma}{\rho})$$


```julia
l31 = simplify(Γ/(-p_ρ + u^2 - Γ*(-e + u^2/2) - (u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))*(u*(2 - Γ) - (u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/ρ)/ρ))
```




\begin{equation*}\frac{2 ρ}{2 e ρ + u^{2} ρ + 2 u \sqrt{ρ \left(p_{ρ} ρ - Γ σ\right)} - 2 σ}\end{equation*}




```julia
 l32 = simplify(Γ*(u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/(ρ*(-p_ρ + u^2 - Γ*(-e + u^2/2) - (u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))*(u*(2 - Γ) - (u*ρ + sqrt(p_ρ*ρ^2 - Γ*ρ*σ))/ρ)/ρ)))
```




\begin{equation*}\frac{2 \left(u ρ + \sqrt{ρ \left(p_{ρ} ρ - Γ σ\right)}\right)}{2 e ρ + u^{2} ρ + 2 u \sqrt{ρ \left(p_{ρ} ρ - Γ σ\right)} - 2 σ}\end{equation*}



$$ l_2 = (1, u+ c, E+uc-\frac{\sigma}{\rho})$$

# 基于原始变量的推导 （not finish）


```julia
T = sympy.Matrix([u ρ 0; 0 u 1/ρ; 0 ρ*a^2-Γ*sxx u])
```




\[\left[ \begin{array}{rrr}u&ρ&0\\0&u&\frac{1}{ρ}\\0&a^{2} ρ - sxx Γ&u\end{array}\right]\]




```julia
A = eigvals(T)
```




\[ \left[ \begin{array}{r}\frac{u ρ + \sqrt{a^{2} ρ^{2} - sxx Γ ρ}}{ρ}\\u\\\frac{u ρ - \sqrt{a^{2} ρ^{2} - sxx Γ ρ}}{ρ}\end{array} \right] \]




```julia
A = eigvecs(T)
```




    3-element Array{Array{Sym,1},1}:
     [1, 0, 0]                                                                                        
     [(u - (u*ρ - sqrt(a^2*ρ^2 - sxx*Γ*ρ))/ρ)^(-2), -1/(ρ*(u - (u*ρ - sqrt(a^2*ρ^2 - sxx*Γ*ρ))/ρ)), 1]
     [(u - (u*ρ + sqrt(a^2*ρ^2 - sxx*Γ*ρ))/ρ)^(-2), -1/(ρ*(u - (u*ρ + sqrt(a^2*ρ^2 - sxx*Γ*ρ))/ρ)), 1]




```julia
A1 = (u - (u*ρ - sqrt(a^2*ρ^2 - sxx*Γ*ρ))/ρ)^(-2)
simplify(A1)
```




\begin{equation*}\frac{ρ}{a^{2} ρ - sxx Γ}\end{equation*}




```julia
A2 = -1/(ρ*(u - (u*ρ - sqrt(a^2*ρ^2 - sxx*Γ*ρ))/ρ))
simplify(A2)
```




\begin{equation*}- \frac{1}{\sqrt{ρ \left(a^{2} ρ - sxx Γ\right)}}\end{equation*}




```julia
A3  = -1/(ρ*(u - (u*ρ + sqrt(a^2*ρ^2 - sxx*Γ*ρ))/ρ))
simplify(A3)
```




\begin{equation*}\frac{1}{\sqrt{ρ \left(a^{2} ρ - sxx Γ\right)}}\end{equation*}




```julia
sympy.hessian(ex, (x,y))
```




\[\left[ \begin{array}{rr}2 y&2 x - 2 y\\2 x - 2 y&- 2 x\end{array}\right]\]


