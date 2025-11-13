import torch
from torch.autograd.functional import vjp
from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fehlberg2 import Fehlberg2
from .fixed_grid import Euler, Midpoint, Heun3, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .dopri8 import Dopri8Solver
from .tsit5 import Tsit5Solver
from .scipy_wrapper import ScipyWrapperODESolver
from .misc import _check_inputs, _flat_to_shape
# from .repara import time_input_repara
from .repara_init import time_input_repara

SOLVERS = {
    "dopri8": Dopri8Solver,
    "dopri5": Dopri5Solver,
    "tsit5": Tsit5Solver,
    "bosh3": Bosh3Solver,
    "fehlberg2": Fehlberg2,
    "adaptive_heun": AdaptiveHeunSolver,
    "euler": Euler,
    "midpoint": Midpoint,
    "heun3": Heun3,
    "rk4": RK4,
    "explicit_adams": AdamsBashforth,
    "implicit_adams": AdamsBashforthMoulton,
    # Backward compatibility: use the same name as before
    "fixed_adams": AdamsBashforthMoulton,
    # ~Backwards compatibility
    "scipy_solver": ScipyWrapperODESolver,
}


@time_input_repara
def odeint(
    func,
    y0,
    t,
    *,
    rtol=1e-7,
    atol=1e-9,
    method=None,
    options=None,
    event_fn=None,
):
    """
    ============================================
    功能描述:常微分方程组(ODE)数值积分求解器
    ============================================

    【数学背景】
    用于求解如下形式的初值问题:
        dy/dt = func(t, y),    y(t0) = y0

    即根据用户定义的函数 func(t, y),在给定时间序列 t 上,
    通过数值方法(如RK4、Dormand-Prince等)求出 y(t)。

    【参数说明】
    ----------
    func : Callable[[Tensor, Tensor], Tensor]
        - 输入为 (t, y)
        - 输出为 dy/dt
        - 表示方程的右侧函数(即微分形式)
        示例:func = lambda t, y: -y + torch.sin(t)

    y0 : Tensor 或 Tuple[Tensor]
        - 初始值,表示 t[0] 时刻的状态向量 y(t0)
        - 可以是标量、多维张量或张量元组

    t : 1D Tensor
        - 时间序列,如 [0, 0.1, 0.2, 1.0]
        - 决定在哪些时间点上求解 ODE
        - 若 t 是递减的,则会自动识别反向积分

    rtol : float,可选
        - 相对误差容忍度 (relative tolerance)
        - 控制积分精度(越小越精确,但计算量增加)
        - 默认值:1e-7

    atol : float,可选
        - 绝对误差容忍度 (absolute tolerance)
        - 控制积分精度的绝对下限
        - 默认值:1e-9

    method : str,可选
        - 指定积分方法名称,如:
          "euler"、"rk4"、"dopri5"、"bosh3" 等
        - 若为 None,则使用默认求解器(通常为高阶RK)

    options : dict,可选
        - 对应 method 的附加参数字典
        - 如 {'step_size': 0.1, 'max_num_steps': 10000}

    event_fn : Callable[[Tensor], Tensor],可选
        - 事件函数,当其输出为0时终止积分(常用于检测碰撞或阈值)
        - 若 event_fn 不为 None,则仅计算直到事件触发的时刻

    【返回值】
    ----------
    若 event_fn 为 None:
        Tensor:形状为 (len(t), *y0.shape),表示每个时间点的解
    若 event_fn 不为空:
        Tuple[Tensor, Tensor]:
            (event_t, solution)
            event_t :事件发生的时间点
            solution:至事件点的积分结果

    【异常】
    ----------
    ValueError:若指定的积分方法不存在于可用求解器表 SOLVERS 中。

    """

    # ==================================================
    # (1)输入检查与标准化
    # ==================================================
    (
        shapes,          # y0 的原始结构信息(若为tuple则记录内部形状)
        func,            # 经过包装后的 func(可能添加了梯度兼容、类型修正)
        y0,              # 转换为扁平化的 Tensor 形式,便于数值操作
        t,               # 确保时间序列为单调升或降
        rtol,            # 相对误差限
        atol,            # 绝对误差限
        method,          # 积分方法名
        options,         # 方法附加选项
        event_fn,        # 事件检测函数(若有)
        t_is_reversed,   # 布尔标志,标识时间是否为反向积分
    ) = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    # ==================================================
    # (2)根据 method 创建对应的求解器对象
    # ==================================================
    # SOLVERS 是一个字典,如:
    # {
    #   "euler": EulerSolver,
    #   "rk4": RungeKutta4,
    #   "dopri5": DormandPrince5,
    #   ...
    # }
    #
    # solver = 具体求解器实例化,例如:
    # solver = RungeKutta4(func=func, y0=y0, rtol=rtol, atol=atol, **options)
    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)

    # ==================================================
    # (3)执行积分(根据是否存在事件函数选择路径)
    # ==================================================
    if event_fn is None:
        # 直接在整个时间序列 t 上进行积分
        solution = solver.integrate(t)
    else:
        # 若存在事件函数,则仅积分到事件发生的时刻(event_fn = 0)
        event_t, solution = solver.integrate_until_event(t[0], event_fn)
        # 将事件时间转换为与输入 t 相同的 Tensor 类型
        event_t = event_t.to(t)
        # 若时间方向反转,则调整符号
        if t_is_reversed:
            event_t = -event_t

    # ==================================================
    # (4)结果重整(若输入为多张量结构,则恢复原形状)
    # ==================================================
    if shapes is not None:
        # 将扁平化结果恢复为原始结构 (len(t), *y0.shape)
        solution = _flat_to_shape(solution, (len(t),), shapes)

    # ==================================================
    # (5)返回结果
    # ==================================================
    if event_fn is None:
        # 无事件函数 → 直接返回积分结果
        return solution
    else:
        # 有事件函数 → 返回事件发生时间与对应解
        return event_t, solution



def odeint_event(
    func, y0, t0, *, event_fn, reverse_time=False, odeint_interface=odeint, **kwargs
):
    """Automatically links up the gradient from the event time."""

    if reverse_time:
        t = torch.cat([t0.reshape(-1), t0.reshape(-1).detach() - 1.0])
    else:
        t = torch.cat([t0.reshape(-1), t0.reshape(-1).detach() + 1.0])

    event_t, solution = odeint_interface(func, y0, t, event_fn=event_fn, **kwargs)

    # Dummy values for rtol, atol, method, and options.
    shapes, _func, _, t, _, _, _, _, event_fn, _ = _check_inputs(
        func, y0, t, 0.0, 0.0, None, None, event_fn, SOLVERS
    )

    if shapes is not None:
        state_t = torch.cat([s[-1].reshape(-1) for s in solution])
    else:
        state_t = solution[-1]

    # Event_fn takes in negated time value if reverse_time is True.
    if reverse_time:
        event_t = -event_t

    event_t, state_t = ImplicitFnGradientRerouting.apply(
        _func, event_fn, event_t, state_t
    )

    # Return the user expected time value.
    if reverse_time:
        event_t = -event_t

    if shapes is not None:
        state_t = _flat_to_shape(state_t, (), shapes)
        solution = tuple(
            torch.cat([s[:-1], s_t[None]], dim=0) for s, s_t in zip(solution, state_t)
        )
    else:
        solution = torch.cat([solution[:-1], state_t[None]], dim=0)

    return event_t, solution


class ImplicitFnGradientRerouting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, event_fn, event_t, state_t):
        """event_t is the solution to event_fn"""
        ctx.func = func
        ctx.event_fn = event_fn
        ctx.save_for_backward(event_t, state_t)
        return event_t.detach(), state_t.detach()

    @staticmethod
    def backward(ctx, grad_t, grad_state):
        func = ctx.func
        event_fn = ctx.event_fn
        event_t, state_t = ctx.saved_tensors

        event_t = event_t.detach().clone().requires_grad_(True)
        state_t = state_t.detach().clone().requires_grad_(True)

        f_val = func(event_t, state_t)

        with torch.enable_grad():
            c, (par_dt, dstate) = vjp(event_fn, (event_t, state_t))

        # Total derivative of event_fn wrt t evaluated at event_t.
        dcdt = par_dt + torch.sum(dstate * f_val)

        # Add the gradient from final state to final time value as if a regular odeint was called.
        grad_t = grad_t + torch.sum(grad_state * f_val)

        dstate = dstate * (-grad_t / (dcdt + 1e-12)).reshape_as(c)

        grad_state = grad_state + dstate

        return None, None, None, grad_state
