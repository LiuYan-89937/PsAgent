# 多模态自动修图 Agent 开发文档

## 1. 项目目标

本项目要做的是一个“看图 + 理解用户表达 + 记住用户偏好”的修图 Agent。

系统需要支持两类使用方式：

1. 用户给出明确修图指令，系统按要求执行。
2. 用户不提供明确指令，系统根据图片内容和用户长期偏好，自动生成一个保守、可解释的修图方案并执行。

这个系统的重点不是“做一个万能大模型入口”，而是做一个职责清晰、可以持续迭代的 Agent 架构。后续无论更换模型、替换工具、增加审核流程，系统主干都不应该被推翻。

## 2. 整体系统架构设计

### 2.1 总体架构

整个系统建议拆成五层：

1. 输入层
   负责接收图片、用户文本、线程信息和用户身份信息。
2. 编排层
   负责组织整个工作流，决定先做什么、后做什么、何时暂停、何时恢复。
3. 理解与规划层
   负责图像分析、指令理解、偏好融合、结构化计划生成。
4. 执行层
   负责真正对图片做编辑，分为确定性执行器、生成式执行器和混合执行器。
5. 记忆与反馈层
   负责读取长期偏好、写入稳定偏好、记录用户接受或拒绝的证据。

当前项目里，核心主干应当围绕 `app/graph/` 展开，而不是先做 API。

### 2.2 为什么要这样拆

这样分层的好处有三点：

1. 稳定。
   简单编辑交给确定性工具，输出更可控。
2. 可扩展。
   后面要加多模型、多工具、多候选图，不需要重写主图。
3. 可解释。
   每一步都能知道输入是什么、输出是什么、为什么进入某条执行分支。

### 2.3 推荐的数据流

推荐整个系统的数据流如下：

```text
输入请求
  -> 读取当前线程上下文
  -> 读取用户长期偏好
  -> 分析图片
  -> 解析用户意图
  -> 生成结构化 edit_plan
  -> 路由执行器
      -> 判断每个 package 是否需要 mask
      -> 需要时再请求阿里云分割
      -> 调用具体工具函数
  -> 评估结果
  -> 如有风险则进入人工确认
  -> 写回长期偏好
  -> 输出最终结果
```

当前阶段最重要的执行原则是：

1. 不是先把整张图统一分割一遍再决定怎么修。
2. 而是先由 Planner 选择工具包。
3. 只有工具包声明“需要 mask”时，才去请求阿里云分割。

### 2.4 当前代码骨架与架构的对应关系

当前仓库中的目录已经基本对应了这个架构：

1. [app/graph/builder.py](/Users/liuyan/Desktop/PsAgent/app/graph/builder.py)
   主图编排入口。
2. [app/graph/state.py](/Users/liuyan/Desktop/PsAgent/app/graph/state.py)
   主状态与核心数据结构。
3. [app/graph/nodes](/Users/liuyan/Desktop/PsAgent/app/graph/nodes)
   各个主图节点。
4. [app/graph/subgraphs](/Users/liuyan/Desktop/PsAgent/app/graph/subgraphs)
   各种执行器子图。
5. [app/memory](/Users/liuyan/Desktop/PsAgent/app/memory)
   长期记忆相关模块。
6. [app/tools](/Users/liuyan/Desktop/PsAgent/app/tools)
   实际图像处理和模型调用的工具层。

后面所有实现都应该围绕这套目录演进，不建议再推翻重来。

### 2.5 当前阶段的范围收敛

根据当前决策，第一阶段不再追求开放域任意内容编辑，而是先把“照片增强”主链做稳。

当前阶段只做：

1. 整体参数调整，例如曝光、对比度、白平衡、去噪、锐化。
2. 局部参数调整，例如脸部提亮、主体提亮、背景轻度压暗。
3. 构图优化，例如轻度裁剪和拉直。
4. 在不改变图片内容的前提下提升照片质感。

当前阶段明确不做：

1. 去掉任意物体。
2. 换背景。
3. 开放文本目标定位。
4. 局部重绘和内容替换。

这个收敛会直接影响架构：

1. 主干执行器以 `deterministic` 为主。
2. 区域能力只需要稳定的人像/主体分割。
3. 生成式执行器先保留骨架，不进入当前主开发序列。

### 2.6 当前阶段的分割层决策

当前项目整体仍然采用“阿里云分割能力 + 本地图像参数工具”的组合，但当前第一批工具包先不依赖分割。

建议这样落地：

1. 人像分割使用阿里云 `SegmentHDBody`
   负责人物主体与背景分离，适合脸部/人物区域相关增强。
2. 主体分割使用阿里云 `SegmentCommonImage` 或 `SegmentHDCommonImage`
   负责主视觉主体与背景分离，适合主体提亮和背景轻度调整。
3. 本地工具负责真正的参数调整
   例如曝光、白平衡、对比度、锐化、裁剪、局部增强。

这样拆的好处是：

1. 当前阶段不需要自部署分割模型。
2. 区域能力和参数处理能力职责清晰。
3. 后续如果要换供应商，只需要替换 `segmentation_tools.py` 的适配层。

补充当前实施顺序：

1. 第一批先落 8 个核心参数工具包，并先实现它们的 `whole_image` 模式，不依赖阿里云分割。
2. 第二批再为其中一部分工具包增加 `person`、`main_subject`、`background` 这类局部 `region` 支持。
3. 也就是说，阿里云分割能力是当前架构中的保留能力，但不是第一批实现的前置条件。

## 3. 核心模块设计

### 3.1 编排层

编排层使用 LangGraph。

这一层负责：

1. 管理节点执行顺序。
2. 在节点间传递 `EditState`。
3. 根据 `edit_plan.executor` 决定执行哪类子图。
4. 挂接 `checkpointer` 管理短期记忆。
5. 挂接 `store` 管理长期记忆。
6. 在必要时通过 `interrupt` 中断执行，等待人工确认。

编排层是整个系统的大脑中枢，但它本身不应该承担具体图像编辑逻辑。

### 3.2 理解与规划层

这一层建议继续拆成三个职责：

1. 图片理解
   判断图片域、场景、主体和问题点。
2. 意图解析
   判断用户是显式修图还是自动修图，并提取限制条件和偏好线索。
3. 计划生成
   把图片分析、用户文本和用户长期偏好融合成结构化 `edit_plan`。

这层的核心输出不是一句自然语言，而是一份稳定的结构化计划。

对当前阶段来说，这层最好再明确成两步：

1. `analyze_image`
   输出“图片事实”，例如是否是人像、是否偏暗、主体是否突出。
2. `build_plan`
   根据这些事实去选择工具包，而不是直接决定底层某个 OpenCV 参数。

### 3.3 执行层

执行层分三类：

1. `deterministic`
   适合曝光、对比度、白平衡、局部提亮、去噪、锐化、裁剪、背景轻度处理。
2. `generative`
   适合去路人、去杂物、换背景、局部重绘、复杂修复。
3. `hybrid`
   适合需要先做局部检测，再做混合编辑的场景。

执行层的职责是“把计划翻译成对图像的具体操作”，而不是“自己决定要怎么修”。

对当前阶段，要再补一个明确判断：

1. 第一批主干执行器完全以 `deterministic` 为主。
2. `hybrid` 主要留给第二批“同一批工具包切换到局部 `region` 模式”时使用。
3. `generative` 继续保留架构位置，但暂时不是当前重点。

### 3.3.1 当前阶段推荐的工具包模式

当前阶段更推荐把执行层理解成“工具包驱动”，而不是“节点里直接调底层工具函数”。

每个工具包只负责一类容易理解的编辑能力，例如：

1. `adjust_exposure`
2. `adjust_highlights_shadows`
3. `adjust_contrast`
4. `adjust_white_balance`
5. `adjust_vibrance_saturation`
6. `crop_and_straighten`
7. `denoise`
8. `sharpen`

每个工具包内部可以包含：

1. 能力说明
2. 输入 schema
3. 是否需要 mask
4. 支持哪些区域
5. 参数兜底逻辑
6. 最终调用哪个具体工具函数

这样设计的好处是：

1. 大模型负责“选包”和“给抽象参数”。
2. 代码负责“补 mask”和“参数适配”。
3. 工具函数只负责真正处理图像。

### 3.3.2 当前工具包结构总结

结合当前讨论，工具包层建议固定成下面这套结构。

#### 设计原则

当前更推荐把“声明能力”和“真正执行”分开。

不要让一个工具包类同时承担：

1. 给 LLM 暴露能力说明
2. 管理执行前依赖
3. 自己偷偷去拉 mask
4. 调 OpenCV / Pillow / 云 API
5. 处理审计和 fallback

更稳的方式是拆成：

1. `PackageSpec`
   纯声明信息，给 Planner、注册表和审计层使用。
2. `ToolPackage`
   真正的执行对象，负责校验、归一化和执行。
3. `PackageRegistry`
   统一注册和导出。
4. `Dispatcher`
   根据工具包声明补齐前置依赖，例如 mask。

#### `PackageSpec`

建议先定义一个纯声明对象，而不是把所有字段都塞在基类上。

建议至少包含：

1. `name`
   工具包唯一标识，例如 `adjust_exposure`。
2. `description`
   给 Planner 和开发者看的简短说明。
3. `supported_regions`
   支持哪些区域，例如 `whole_image`、`person`、`background`。
4. `mask_policy`
   是否需要 mask，建议使用：
   `none / optional / required`
5. `supported_domains`
   适用于哪些图片域，例如 `portrait`、`general`。
6. `risk_level`
   风险等级，给审核和风控使用。
7. `default_params`
   默认参数，用来兜底和归一化。

之所以推荐 `mask_policy` 而不是简单的 `requires_mask`，是因为：

1. 有些包完全不需要 mask
2. 有些包全局和局部都能做，mask 是可选项
3. 有些包天生就是局部增强，mask 是必需项

#### 工具包生命周期方法

每个工具包建议统一实现下面这些方法：

1. `get_llm_schema()`
   导出给 LLM 的简化 schema。
2. `validate(operation, context)`
   校验模型给出的调用请求是否合法。
3. `resolve_requirements(operation, context)`
   判断是否需要 mask、是否需要主体信息等前置依赖。
4. `normalize(operation, context)`
   把抽象参数转成更稳定的内部参数。
5. `execute(operation, context)`
   真正执行图像处理。
6. `fallback(error, operation, context)`
   执行失败时做降级、跳过或请求审核。

这里建议再加一个重要约束：

1. 工具包自己不直接去调用分割接口拿 mask
2. 工具包只通过 `resolve_requirements()` 声明自己需要什么
3. 真正去补 mask 的动作由 `Dispatcher` 或执行器完成

这样做的好处是：

1. 分割逻辑不会散落在每个包里
2. 更容易统一审计和替换供应商
3. 每个包更容易单元测试

#### `PackageRegistry`

建议不要只靠类继承，还要有统一注册表。

注册表负责：

1. 注册所有启用中的工具包。
2. 按 `name` 查找工具包。
3. 按 `domain`、`region`、`risk_level` 过滤工具包。
4. 导出“给 LLM 看的能力清单”。

当前阶段建议先采用显式注册，不做动态扫描：

1. `registry.register(AdjustExposurePackage())`
2. `registry.register(SubjectLightBalancePackage())`
3. `registry.register(BackgroundToneBalancePackage())`

这样更稳，也更方便调试和审计。

当前阶段建议注册的首批工具包包括：

1. `adjust_exposure`
2. `adjust_highlights_shadows`
3. `adjust_contrast`
4. `adjust_white_balance`
5. `adjust_vibrance_saturation`
6. `crop_and_straighten`
7. `denoise`
8. `sharpen`

这 8 个工具包组成当前第一批落地范围，优先目标是先把它们在 `whole_image` 模式下跑稳。

第二批不是再新增一套“局部工具包”，而是为其中一部分已有工具包补充局部 `region` 支持，例如：

1. `adjust_exposure` 支持 `person`、`main_subject`、`background`
2. `adjust_contrast` 支持 `main_subject`、`background`
3. `adjust_vibrance_saturation` 支持 `main_subject`、`background`
4. `denoise`、`sharpen` 在局部模式下使用更保守参数

#### 工具包如何绑定到 LLM

当前阶段不建议把每个工具包都直接暴露成 `@tool` 给模型。

更稳的方式是：

1. `PackageRegistry` 统一导出工具包清单和 schema。
2. `build_plan` 把这份能力清单提供给 LLM。
3. LLM 只输出结构化 `edit_plan.operations`。
4. 运行时根据 `op` 去注册表里找到对应工具包。
5. 由执行器决定是否先补 mask，再调用工具包执行。

也就是说：

1. LLM 负责“选包”。
2. 代码负责“执行包”。
3. 监控和审计发生在代码执行层，而不是依赖 `@tool` 自动追踪。

#### `OperationContext`

建议所有工具包执行时都接收统一上下文，而不是每个函数自己散着接参。

上下文里建议至少包含：

1. 当前输入图路径
2. `image_analysis`
3. `retrieved_prefs`
4. 已经拿到的 `masks`
5. `thread_id`
6. 执行过程中的审计对象

建议后续把它设计成统一上下文对象，而不是让每个工具函数自己定义一套参数签名。

#### `PackageResult`

建议所有工具包返回统一结果结构，方便评估层和审计层消费。

建议至少包含：

1. `ok`
2. `package`
3. `output_image`
4. `applied_params`
5. `warnings`
6. `artifacts`
7. `fallback_used`
8. `error`

统一结果结构的意义在于：

1. `evaluate_result` 不需要理解每个工具包的内部差异。
2. 审计日志可以统一记录。
3. 后续接 LangSmith 或自定义 trace 更容易。

#### 当前阶段的最小落地建议

如果按当前项目节奏推进，最推荐的落地顺序是：

1. 先定义 `PackageSpec`
2. 再定义 `EditOperation`
3. 再定义 `OperationContext` 和 `PackageResult`
4. 再实现 `ToolPackage` 抽象基类
5. 再实现 `PackageRegistry`
6. 最后先落一个最简单的包，例如 `adjust_exposure`

### 3.4 记忆层

记忆层建议分短期和长期两类：

1. 短期记忆
   保存当前线程的上下文、当前轮分析结果和当前计划。
2. 长期记忆
   保存用户稳定偏好和历史反馈证据。

短期记忆解决“这一轮对话上下文是什么”，长期记忆解决“这个用户长期喜欢什么风格”。

### 3.5 评估与审核层

系统不能只会修图，还要会判断这次修图结果是否可信。

这一层负责：

1. 结果评价
   看结果是否符合计划、是否过度修图、是否有明显风险。
2. 审核控制
   对高风险场景发起人工确认。
3. 恢复执行
   用户确认后恢复 LangGraph 流程。

## 4. 当前骨架中最重要的数据结构

当前骨架里的核心数据结构已经放在 [app/graph/state.py](/Users/liuyan/Desktop/PsAgent/app/graph/state.py)。

### 4.1 `EditOperation`

它表示一次具体编辑动作。

在当前阶段，更建议把它理解成一次“工具包级操作请求”，而不是直接面向底层图像库的调用。

建议它长期稳定包含这些字段：

1. `op`
   工具包标识，例如 `adjust_exposure`、`adjust_vibrance_saturation`。
2. `region`
   作用区域，例如 `whole_image`、`person`、`main_subject`、`background`。
3. `strength`
   强度。
4. `params`
   额外参数。

对当前阶段来说，`region` 建议先控制在有限稳定集合内，例如：

1. `whole_image`
2. `face`
3. `person`
4. `main_subject`
5. `background`

这些区域由 [app/tools/segmentation_tools.py](/Users/liuyan/Desktop/PsAgent/app/tools/segmentation_tools.py) 统一向阿里云请求分割结果，不在 Planner 层直接绑定云 API 细节。

这里最关键的边界是：

1. `EditOperation` 只表达“想做什么”。
2. 它不直接携带阿里云 API 参数。
3. 它也不直接携带 OpenCV/Pillow 的低层参数。
4. 是否需要 mask，由工具包定义决定，而不是由 Planner 直接拼底层调用。

### 4.2 `EditPlan`

它是整个系统里最重要的中间产物。

建议长期稳定包含这些字段：

1. `mode`
2. `domain`
3. `executor`
4. `preserve`
5. `operations`
6. `should_write_memory`
7. `memory_candidates`
8. `needs_confirmation`

一句话理解：

`EditPlan` 决定了系统“为什么这样修、要修哪里、该用哪种执行器、要不要触发审核”。

对当前阶段来说，`EditPlan.operations` 更适合被理解成：

1. 一组工具包调用请求
2. 每个请求带有区域、强度和约束
3. 后续执行器再根据工具包能力决定是否先做分割

### 4.3 `PreferenceMemory`

它表示一条长期偏好记录。

建议保留的字段包括：

1. `user_id`
2. `domain`
3. `key`
4. `value`
5. `confidence`
6. `source`
7. `evidence_count`
8. `last_updated_at`

### 4.4 `EditState`

它是 LangGraph 主图的状态容器。

当前骨架已经包含以下关键字段：

1. `messages`
2. `user_id`
3. `thread_id`
4. `input_images`
5. `mode`
6. `image_analysis`
7. `retrieved_prefs`
8. `edit_plan`
9. `masks`
10. `candidate_outputs`
11. `eval_report`
12. `selected_output`
13. `memory_write_candidates`
14. `approval_required`
15. `approval_payload`

后续建议继续补充：

1. `instruction`
2. `request_intent`
3. `plan_summary`
4. `execution_trace`
5. `result_status`
6. `review_decision`

## 5. LangGraph 主图设计

主图的骨架已经预留在 [app/graph/builder.py](/Users/liuyan/Desktop/PsAgent/app/graph/builder.py)。

### 5.1 主图节点

当前建议的主图节点是：

1. `load_context`
2. `analyze_image`
3. `parse_request`
4. `build_plan`
5. `route_executor`
6. `execute_deterministic`
7. `execute_generative`
8. `execute_hybrid`
9. `evaluate_result`
10. `human_review`
11. `update_memory`

### 5.2 每个节点应该做什么

#### `load_context`

要做的事：

1. 读取当前线程上下文。
2. 读取用户长期偏好。
3. 整理成后续节点可直接使用的上下文。

当前重点：

1. 先定义输入输出格式。
2. 先返回空结构也可以。
3. 后续再逐步接入真实 store。

#### `analyze_image`

要做的事：

1. 判断图片域。
2. 识别场景类型。
3. 识别主体。
4. 识别问题点，例如逆光、偏暗、噪点、背景杂乱。

当前重点：

1. 先输出一个稳定字典。
2. 先别追求识别模型复杂度。
3. 先让后续节点有可消费的分析结果。

补充说明：

1. 当前阶段不要求它输出开放域实例描述。
2. 当前阶段只要能支撑“人像图”与“主体图”的后续规划即可。
3. 更精确的区域 mask 由后续 `segmentation_tools.py` 在执行前向阿里云请求。
4. 它的职责是输出“图像事实”，不是直接决定调用哪个底层函数。

建议它的来源是：

1. 规则分析
   例如亮度、对比度、噪点、清晰度、构图中心性。
2. 多模态模型理解
   例如图片域、场景标签、主体类型、审美问题标签。

也就是说，`analyze_image` 不应该只让大模型随意输出一堆自然语言，而应该输出结构化事实。

#### `parse_request`

要做的事：

1. 理解用户文本。
2. 判断是显式模式还是自动模式。
3. 提取限制条件。
4. 提取可写入记忆的偏好信号。

当前重点：

1. 先把模式判断做对。
2. 再逐步补意图解析细节。

#### `build_plan`

要做的事：

1. 合并图片分析结果。
2. 合并用户文本。
3. 合并长期偏好。
4. 选择合适的工具包。
5. 生成结构化计划。

当前重点：

1. 先保证 `edit_plan` 结构稳定。
2. 不要一开始就做太复杂的策略。
3. 不要在这一步直接决定底层图像库参数。

更准确地说，这一步应该产出的是：

1. 要调用哪些工具包
2. 每个工具包的作用区域
3. 每个工具包的抽象强度
4. 每个工具包是否需要后续补区域信息

#### `route_executor`

要做的事：

1. 判断执行器类型。
2. 检查每个工具包是否需要 mask。
3. 准备后续执行所需的信息。
4. 预判是否需要人工审核。

当前重点：

1. 路由规则先写死。
2. 先把“需要 mask 才去分割”这条规则做稳。
3. 后面再考虑模型辅助决策。

#### `execute_*`

要做的事：

1. 根据计划逐个执行工具包。
2. 某个工具包需要 mask 时，先调用分割层。
3. 把抽象参数转换成具体工具参数。
4. 统一输出候选结果格式。

当前重点：

1. 先统一返回格式。
2. 先别在 Graph 里处理复杂图像逻辑。
3. Graph 里只做调度，不直接写底层图像处理细节。

#### `evaluate_result`

要做的事：

1. 检查结果是否符合计划。
2. 检查结果是否有伪影。
3. 检查是否有过度修图。
4. 决定是否需要审核。

当前重点：

1. 先做规则评价。
2. 后面再接 Critic 模型。

#### `human_review`

要做的事：

1. 对高风险编辑触发中断。
2. 接收用户确认。
3. 恢复执行。

当前重点：

1. 先定义审核触发条件。
2. 先打通 interrupt / resume 流程。

#### `update_memory`

要做的事：

1. 从本轮结果里抽取稳定偏好。
2. 写回长期记忆。
3. 更新证据计数和置信度。

当前重点：

1. 先支持显式偏好写入。
2. 后面再支持行为证据总结。

## 6. 执行层设计建议

### 6.1 确定性执行器先做什么

建议第一批先做这 8 个工具包：

1. `adjust_exposure`
2. `adjust_highlights_shadows`
3. `adjust_contrast`
4. `adjust_white_balance`
5. `adjust_vibrance_saturation`
6. `crop_and_straighten`
7. `denoise`
8. `sharpen`

原因：

1. 这些操作最常用。
2. 这些操作最稳定。
3. 这些操作最适合先跑通主链。

建议对应模块：

1. [app/tools/opencv_tools.py](/Users/liuyan/Desktop/PsAgent/app/tools/opencv_tools.py)
2. [app/tools/image_ops.py](/Users/liuyan/Desktop/PsAgent/app/tools/image_ops.py)
3. 参数归一化层
4. 单元测试

其中：

1. `opencv_tools.py` 和 `image_ops.py` 负责真正的像素参数调整。
2. 参数归一化层负责把抽象强度映射成稳定的内部参数。
3. 单元测试负责验证参数可行性和结果可重复性。

当前阶段更推荐的执行方式是：

1. `build_plan` 先选择工具包
2. 参数归一化层先把抽象参数变成可执行参数
3. 执行器直接调 OpenCV/Pillow 具体函数
4. 局部 `region` 的测试链路可以先接入 `segmentation_tools.py`
5. 正式执行时，仍然保持“外层统一准备 mask，工具包只消费 mask”

当前 `adjust_exposure` 的测试已经收敛成两段：

1. 用 `tests/` 下的真人图片直接测 `whole_image` 曝光
2. 联网调用阿里云 `SegmentHDBody + RefineMask` 实时生成二值人像 mask，再测 `person` 局部曝光

### 6.2 生成式执行器后做什么

建议第二阶段再做这些：

1. `remove_object`
2. `replace_background`
3. `inpaint_region`

原因：

1. 生成式编辑需要更多风控。
2. 生成式编辑更容易引入伪影。
3. 先让 Graph 主链稳定更重要。

建议对应模块：

[app/tools/image_edit_model.py](/Users/liuyan/Desktop/PsAgent/app/tools/image_edit_model.py)

### 6.3 Hybrid 为什么单独保留

真实产品里，很多场景既不是纯参数调节，也不是纯生成式改图。

例如：

1. 先识别脸部，再局部提亮。
2. 先抠出主体或背景，再做轻度区域调色。
3. 先做基础曝光调整，再对主体做保守增强。

因此 `hybrid` 不是“以后再说”，而是应该从一开始就在架构中保留。

对当前阶段来说，`hybrid` 的主要含义是：

1. 不是统一先分割整张图。
2. 而是某个工具包要求局部区域时，再请求阿里云分割结果。
3. 再在分割区域上做本地确定性调整。
4. 不做内容生成和内容替换。

但按当前实施顺序：

1. 第一批先只实现这 8 个工具包的 `whole_image` 模式，因此先不进入 `hybrid`
2. 第二批当同一批工具包开始支持局部 `region` 时，再正式启用 `hybrid`

## 7. 记忆系统设计建议

### 7.1 短期记忆

短期记忆保存的是“当前线程内有效”的内容。

建议保存：

1. 当前图片
2. 当前输入文本
3. 当前分析结果
4. 当前计划
5. 当前审核状态
6. 当前候选图

短期记忆的目标是保证流程上下文不断档，而不是做用户画像。

### 7.2 长期记忆

长期记忆保存的是“跨会话稳定有效”的偏好。

建议使用两层结构：

1. Profile
   保存稳定偏好摘要。
2. Event
   保存每次反馈和行为证据。

建议先记录这些类型：

1. 用户明确说不要磨皮。
2. 用户明确说喜欢偏暖或偏冷。
3. 用户明确说不要太艳。
4. 用户明确说保留真实感。
5. 用户多次接受某种修图风格。

### 7.3 偏好写入规则

以下情况建议写入长期记忆：

1. 用户显式表达。
2. 同类偏好多次重复出现。
3. 用户连续接受某种风格。
4. 用户明确拒绝某种风格。

以下情况不要直接写入：

1. 单张图片的临时特效。
2. 非稳定的实验性风格。
3. 模型的单次猜测。

## 8. 自动修图模式设计

自动修图是一个增强能力，不建议第一版就做得很激进。

### 8.1 自动模式的正确目标

目标不是“让系统自由发挥”，而是：

1. 给出保守计划。
2. 遵守长期偏好。
3. 遵守保真约束。
4. 结果可解释。

### 8.2 人像自动策略建议

第一版建议只允许：

1. 脸部轻微提亮
2. 肤色轻微统一
3. 背景轻度清理
4. 眼部轻度锐化
5. 保持自然，不做明显磨皮

### 8.3 风景自动策略建议

第一版建议只允许：

1. 阴影提亮
2. 高光回收
3. 轻度去雾
4. 轻度对比增强
5. 适度裁剪

### 8.4 自动模式限制

第一版建议加上以下硬约束：

1. 强度更低
2. 默认禁止高风险生成式编辑
3. 如涉及身份风险，必须人工确认
4. 输出时要附一句解释摘要

## 9. 风控与可观测性

### 9.1 风控

需要重点控制的风险包括：

1. 人像身份变化
2. 过度磨皮
3. 过度饱和
4. 明显伪影
5. 自动模式下误改主体

建议默认进入人工确认的场景：

1. 换背景
2. 去除大面积背景元素
3. 大范围局部重绘
4. 人脸附近复杂生成式编辑

### 9.2 可观测性

建议从一开始就在服务层预留审计和追踪能力：

1. 记录每个节点耗时
2. 记录节点输入输出摘要
3. 记录执行器选择结果
4. 记录计划版本
5. 记录用户反馈
6. 记录记忆写入结果

对应骨架已经预留在：

[app/services/audit.py](/Users/liuyan/Desktop/PsAgent/app/services/audit.py)

## 10. 详细开发步骤

下面这部分是最重要的，建议你后续开发就按这个顺序推进。

### 第一步：先把状态模型补完整

你要做什么：

1. 明确 `EditState` 的字段边界。
2. 明确哪些字段是输入层负责提供。
3. 明确哪些字段由各节点写入。
4. 明确哪些字段是中间产物，哪些字段是最终输出。

为什么先做这一步：

1. 状态模型一旦不稳定，后面每个节点都要反复改。
2. 先把数据流固定，后面才容易写节点。

重点文件：

[app/graph/state.py](/Users/liuyan/Desktop/PsAgent/app/graph/state.py)

完成标准：

1. 你能写出一份最小输入状态样例。
2. 你能写出一份最小输出状态样例。
3. 所有关键中间字段已经有明确名字。

### 第二步：固定主图结构

你要做什么：

1. 固定节点顺序。
2. 固定条件边。
3. 固定执行器路由函数。
4. 固定审核分支。

为什么先做这一步：

1. Graph 结构是后续所有实现的主干。
2. 主图不稳定，后面每补一个节点都容易返工。

重点文件：

[app/graph/builder.py](/Users/liuyan/Desktop/PsAgent/app/graph/builder.py)

完成标准：

1. 主图节点顺序已经固定。
2. 条件路由已经清晰。
3. 后续只改节点内容，不改整体流程。

### 第三步：实现 `analyze_image`

你要做什么：

1. 先定义图像分析输出格式。
2. 先支持最基础的 domain 判断。
3. 先支持最基础的问题标签。

第一版建议先输出：

1. `domain`
2. `scene_tags`
3. `issues`
4. `subjects`
5. `segmentation_hints`

为什么先做这一步：

1. `build_plan` 和 `parse_request` 都会依赖分析结果。

完成标准：

1. 对任意输入图片都能给出一份最小分析结果。
2. 后续节点不需要猜图片类型。

### 第四步：实现 `parse_request`

你要做什么：

1. 判断显式模式还是自动模式。
2. 提取明确操作意图。
3. 提取用户限制条件。
4. 提取可写入长期记忆的显式偏好。

第一版建议先支持识别：

1. 提亮
2. 白平衡
3. 对比度
4. 去噪
5. 锐化
6. 裁剪
7. 脸部提亮
8. 主体提亮
9. 背景压暗
10. 不要磨皮
11. 保留真实感

完成标准：

1. 文本输入能稳定映射到基础意图结构。
2. 自动模式和显式模式判断正确。

### 第五步：实现 `build_plan`

你要做什么：

1. 把图片分析结果、用户文本、用户偏好融合起来。
2. 生成结构化 `edit_plan`。
3. 生成 `memory_write_candidates`。

这一步最关键的事情：

1. 固定 `EditPlan` 的 schema。
2. 不要返回松散自然语言。
3. 不要把路由逻辑混进执行器里。

这一步最好进一步明确为：

1. 先根据图像事实和用户意图选择工具包
2. 再为每个工具包生成区域、强度和约束
3. 不在这一步直接生成底层工具参数

完成标准：

1. 对任意输入都能生成一份可消费的计划。
2. 计划中明确写出 `executor`。

### 第六步：实现 `route_executor`

你要做什么：

1. 基于 `edit_plan` 判断走哪个执行器。
2. 检查每个工具包是否需要 mask。
3. 准备执行前的额外信息。
4. 判断是否需要人工确认。

第一版建议直接写规则：

1. 第一批这 8 个工具包在 `whole_image` 模式下全部走 `deterministic`
2. 第二批当同一批工具包请求 `person`、`main_subject`、`background` 等局部 `region` 时走 `hybrid`
3. 当前阶段默认不把请求送去 `generative`

完成标准：

1. 计划和执行器之间的映射关系稳定。
2. 评估和审核所需的信息也能提前准备好。

### 第七步：实现确定性执行器

你要做什么：

1. 先实现基础工具包执行逻辑
2. 先让它能处理最基础的图像调节
3. 先统一输出结果结构

建议先实现的操作：

1. 曝光
2. 高光阴影平衡
3. 对比度
4. 白平衡
5. 自然饱和度 / 饱和度
6. 裁剪与拉直
7. 去噪
8. 锐化

为什么它优先：

1. 最稳定
2. 最常用
3. 最容易验证主链正确性

实现时建议采用统一模式：

1. 一个操作先映射到一个工具包
2. 参数归一化层先生成稳定内部参数
3. 真正的工具函数只接收已经整理好的入参

### 第八步：实现 `evaluate_result`

你要做什么：

1. 检查结果是否符合计划。
2. 检查是否有明显风险。
3. 决定是否进入人工确认。

第一版可以先做规则评估，不一定马上上 Critic 模型。

重点判断：

1. 是否有结果输出
2. 是否有高风险操作
3. 是否违反用户的“保留真实”“不要磨皮”等约束

### 第九步：实现长期记忆读取

你要做什么：

1. 在 `memory/retriever.py` 中实现按 domain 读取偏好。
2. 在 `load_context` 中接入偏好读取。
3. 保证 Planner 能拿到最相关的一小部分偏好。

第一版建议只支持：

1. 精确 domain 过滤
2. 简单 profile 读取

为什么不要一开始做太复杂：

1. 先验证“偏好是否真的影响计划”更重要。

### 第十步：实现长期记忆写入

你要做什么：

1. 从用户显式反馈中抽取稳定偏好。
2. 在 `update_memory` 中写入 store。
3. 记录证据次数和更新时间。

第一版先支持：

1. 显式偏好写入
2. 负向偏好写入
3. 简单置信度更新

### 第十一步：实现生成式执行器

你要做什么：

1. 在 `app/tools/image_edit_model.py` 中定义模型调用接口。
2. 在 `execute_generative` 中接入生成式编辑。
3. 保证输出格式与其他执行器一致。

建议先只做一个高价值场景：

1. 去路人
2. 或去杂物

这样更容易验证效果。

但按当前阶段范围，这一条先只保留骨架和接口，不进入近期主开发顺序。

### 第十二步：实现 Hybrid 执行器

你要做什么：

1. 把阿里云分割、局部区域解析、确定性处理串起来。
2. 统一输入输出格式。
3. 保证不会破坏主图结构。

建议后做的原因：

1. 当前阶段的 Hybrid 依赖分割层和确定性工具都先稳定。

### 第十三步：实现自动修图模式

你要做什么：

1. 在 `parse_request` 中补自动模式判断。
2. 在 `build_plan` 中加入自动模式的默认策略。
3. 引入偏好上限和强度约束。

建议第一版只支持：

1. 人像自动修图
2. 主体照片自动修图

### 第十四步：实现人工审核

你要做什么：

1. 在 `human_review` 中接入 `interrupt`
2. 设计恢复后的状态流转
3. 确定哪些编辑必须审核

建议第一批需要审核的场景：

1. 去路人
2. 换背景
3. 大范围局部重绘

### 第十五步：最后再接 API 和服务层

你要做什么：

1. 在 Graph 输入输出契约稳定之后，再接 API。
2. 让 API 只负责参数转换，不负责业务逻辑。
3. 再接任务存储、资产存储和审计。

为什么最后做：

1. 否则前面 Graph 一改，API 也跟着改。
2. 先把核心工作流跑稳更重要。

## 11. 推荐目录结构说明

```text
app/
  api/                  # 暂时预留，不是当前重点
  graph/                # LangGraph 主图、状态、节点、子图
  memory/               # 长期偏好读写与提取
  tools/                # 图像处理和模型调用封装
  services/             # 审计、任务、资产等服务层
  prompts/              # Planner / Critic / Memory prompts
docs/
  multimodal_photo_agent_dev.md
  sequential_development_plan.md
```

## 12. 当前阶段建议

如果从现在开始正式写代码，最推荐的顺序是：

1. 先补 `state.py`
2. 再补 `builder.py`
3. 再补 `analyze_image`
4. 再补 `parse_request`
5. 再补 `build_plan`
6. 再补工具包协议与注册表
7. 再补参数归一化层
8. 再补 `execute_deterministic`
9. 再补第一批 8 个工具包的单元测试
10. 再补 `evaluate_result`
11. 再补 `memory/retriever.py`
12. 再补 `update_memory`
13. 再补 `route_executor`
14. 再补 `segmentation_tools.py`
15. 再补 `execute_hybrid`
16. 再补 `execute_generative`
17. 再补 `human_review`
18. 最后再接 API

一句话总结：

先把 LangGraph 主图做成稳定的“理解 -> 规划 -> 选包 -> 参数归一化 -> 执行 -> 评估 -> 记忆”骨架，先跑通第一批 8 个工具包的 `whole_image` 模式，再为其中一部分增加局部 `region` 与分割能力。
