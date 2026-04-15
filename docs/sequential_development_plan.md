# 多模态自动修图 Agent 顺序开发计划

## 1. 这份文档怎么用

这份文档不讲系统架构原理，专门讲开发顺序。

你可以把它理解成一份“施工手册”：

1. 先做什么
2. 后做什么
3. 每一步要改哪些文件
4. 每一步完成后应该达到什么状态
5. 每一步常见的坑是什么

## 2. 总体开发原则

整个项目建议遵守下面这几个顺序原则：

1. 先让 Graph 主链稳定，再追求模型效果。
2. 先做显式修图，再做自动修图。
3. 先做工具包协议、参数归一化和确定性执行器，再做阿里云分割接入。
4. 先做长期偏好的最小读写，再做复杂偏好召回。
5. 先做规则判断，再做模型增强。
6. 先做核心编排，再做 API。

补充当前阶段范围：

1. 当前阶段只做人像分割、主体分割和照片参数增强。
2. 当前阶段不做开放域实例删除、背景替换和内容重绘。

## 3. 阶段 0：固定主干骨架

### 目标

把项目的主干结构固定下来，避免后面反复改目录和状态模型。

### 你要做什么

1. 确认 `app/graph/` 是主战场。
2. 确认 `app/memory/` 只负责长期偏好。
3. 确认 `app/tools/` 只负责图像工具和模型调用。
4. 确认 `app/services/` 只放服务层能力。
5. 确认当前阶段 API 目录只是预留。

### 必改文件

1. [app/graph/state.py](/Users/liuyan/Desktop/PsAgent/app/graph/state.py)
2. [app/graph/builder.py](/Users/liuyan/Desktop/PsAgent/app/graph/builder.py)

### 完成标准

1. 目录职责清晰。
2. 主图节点名字固定。
3. 状态字段基础框架稳定。

### 常见问题

1. 一开始就把 API、服务层、前端交互一起做。
2. 状态字段还没想清楚就开始堆节点逻辑。

## 4. 阶段 1：把 `EditState` 做完整

### 目标

把主图中所有关键中间产物的状态字段补齐。

### 你要做什么

1. 明确哪些字段是输入字段。
2. 明确哪些字段由节点逐步写入。
3. 明确哪些字段是最终输出。
4. 明确哪些字段是中断恢复所必须的。

### 建议优先确认的字段

1. `user_id`
2. `thread_id`
3. `input_images`
4. `mode`
5. `image_analysis`
6. `retrieved_prefs`
7. `edit_plan`
8. `candidate_outputs`
9. `eval_report`
10. `selected_output`
11. `memory_write_candidates`
12. `approval_required`
13. `approval_payload`

### 文件

[app/graph/state.py](/Users/liuyan/Desktop/PsAgent/app/graph/state.py)

### 完成标准

1. 你能手写一份最小输入状态。
2. 你能手写一份完整输出状态。
3. 每个节点需要改哪些字段已经明确。

## 5. 阶段 2：把主图顺序固定

### 目标

让 Graph 结构不再摇摆。

### 你要做什么

1. 固定从 `START` 到 `END` 的主链。
2. 固定执行器条件路由。
3. 固定审核条件路由。
4. 明确每条边为什么存在。

### 文件

[app/graph/builder.py](/Users/liuyan/Desktop/PsAgent/app/graph/builder.py)

### 完成标准

1. 主图结构不再频繁调整。
2. 后续开发主要聚焦节点内容。

### 常见问题

1. 节点还没实现就频繁改图结构。
2. 一个节点做太多职责，导致 Graph 很难维护。

## 6. 阶段 3：实现图片分析节点

### 目标

让系统先能理解“这是一张什么图，大概有什么问题”。

### 你要做什么

1. 定义 `image_analysis` 输出结构。
2. 支持基础 domain 判断。
3. 支持基础问题标签。
4. 支持基础主体标签。

### 第一版建议输出

1. `domain`
2. `scene_tags`
3. `issues`
4. `subjects`
5. `segmentation_hints`

### 文件

[app/graph/nodes/analyze_image.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/analyze_image.py)

### 完成标准

1. 每张图都能给出最小分析结果。
2. 下游节点不再需要重复猜图像类型。

## 7. 阶段 4：实现意图解析节点

### 目标

让系统先能理解用户到底想让它做什么。

### 你要做什么

1. 判断显式修图还是自动修图。
2. 识别基础操作意图。
3. 识别限制条件。
4. 识别可能的长期偏好信号。

### 第一版建议先识别的表达

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
11. 自然一点

### 文件

[app/graph/nodes/parse_request.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/parse_request.py)

### 完成标准

1. 文本能映射到最小意图结构。
2. 自动模式判断准确。

## 8. 阶段 5：实现计划生成节点

### 目标

让系统输出稳定的结构化 `edit_plan`。

### 你要做什么

1. 合并图片分析结果。
2. 合并用户文本意图。
3. 合并长期偏好。
4. 决定执行器类型。
5. 生成编辑操作列表。
6. 生成记忆候选项。

### 重点

这一步是整个系统的核心，不要偷懒用自然语言替代结构化计划。

这一步最好明确成：

1. 先选择工具包
2. 再生成每个工具包的区域、强度和参数
3. 不要在这里直接拼接底层图像库参数

### 文件

1. [app/graph/nodes/build_plan.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/build_plan.py)
2. [app/graph/state.py](/Users/liuyan/Desktop/PsAgent/app/graph/state.py)

### 完成标准

1. 每次请求都能产出 `edit_plan`。
2. `edit_plan` 可以直接被执行器消费。

## 9. 阶段 6：实现路由节点

### 目标

让计划真正决定执行路径。

### 你要做什么

1. 读取 `edit_plan.executor`
2. 第一批先按全局工具包处理，不要求准备阿里云分割结果
3. 判断是否需要审核
4. 把执行前的信息整理好

### 文件

[app/graph/nodes/route_executor.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/route_executor.py)

### 完成标准

1. 三类执行器路由稳定。
2. 高风险任务会被正确标记。

### 当前阶段的额外要求

1. 需要 `face`、`person`、`main_subject`、`background` 这类区域时，不在节点里直接拼云 API 请求。
2. 节点只负责决定“要不要区域”，真正调用封装到 `segmentation_tools.py`。

## 10. 阶段 7：先做工具包协议与注册表

### 目标

让 Planner 和执行层之间先有一个稳定的中间协议。

### 你要做什么

1. 定义工具包的唯一标识。
2. 定义每个工具包支持的区域类型。
3. 定义每个工具包的 `mask_policy`。
4. 定义每个工具包的输入 schema 和默认兜底参数。
5. 定义工具包统一基类。
6. 定义注册表和统一导出方式。
7. 明确由 Dispatcher 统一补 mask，而不是包自己去拿。

### 当前阶段建议先注册并优先落地的工具包

1. `adjust_exposure`
2. `adjust_highlights_shadows`
3. `adjust_contrast`
4. `adjust_white_balance`
5. `adjust_vibrance_saturation`
6. `crop_and_straighten`
7. `denoise`
8. `sharpen`

这 8 个组成第一批工具包，优先目标是先跑通它们在 `whole_image` 模式下的参数增强链。

第二批不是新增一套局部工具包，而是为其中一部分已有工具包增加局部 `region` 支持，例如：

1. `adjust_exposure`
2. `adjust_contrast`
3. `adjust_vibrance_saturation`
4. `denoise`
5. `sharpen`

### 这一阶段建议同时定下来的结构

1. `PackageSpec`
   纯声明对象，负责描述工具包能力。
2. `ToolPackage`
   统一基类，负责声明能力和生命周期方法。
3. `PackageRegistry`
   统一注册、发现、过滤和导出工具包。
4. `OperationContext`
   统一执行上下文。
5. `PackageResult`
   统一执行结果。

### 基类建议至少包含的能力

1. `name`
2. `description`
3. `supported_regions`
4. `mask_policy`
5. `supported_domains`
6. `risk_level`
7. `default_params`

### 基类建议至少包含的方法

1. `get_llm_schema()`
2. `validate()`
3. `resolve_requirements()`
4. `normalize()`
5. `execute()`
6. `fallback()`

### 关于 LLM 绑定方式

这一阶段建议明确采用：

1. 不把每个工具包直接暴露成 `@tool`
2. 而是由注册表统一导出 schema 给 Planner
3. 让 LLM 输出 `edit_plan.operations`
4. 运行时再根据 `op` 查表执行

### 这一阶段必须明确的执行边界

1. 工具包负责声明自己需要什么
2. Dispatcher 负责准备 mask、主体信息等前置依赖
3. 工具包执行时只消费已经准备好的上下文
4. 不允许每个工具包自己偷偷去请求阿里云分割

### 文件

1. [app/graph/state.py](/Users/liuyan/Desktop/PsAgent/app/graph/state.py)
2. [app/graph/nodes/build_plan.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/build_plan.py)
3. [app/graph/nodes/route_executor.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/route_executor.py)
4. [app/tools](/Users/liuyan/Desktop/PsAgent/app/tools)

### 完成标准

1. Planner 能输出稳定的工具包级操作请求。
2. 执行层能根据工具包声明判断是否需要 mask。
3. 所有工具包都能通过统一注册表被发现。
4. 工具包层已经有统一输入输出协议。
5. 已经明确“声明层”和“执行层”的分工。

## 11. 阶段 8：先做参数归一化层

### 目标

先验证抽象强度是否能稳定映射成真实可执行参数。

### 你要做什么

1. 定义 `strength` 的统一范围。
2. 定义每个工具包的默认值和边界裁剪规则。
3. 定义抽象参数到内部参数的映射。
4. 确保同一输入下结果可重复。

### 当前阶段优先覆盖的工具包

1. `adjust_exposure`
2. `adjust_highlights_shadows`
3. `adjust_contrast`
4. `adjust_white_balance`
5. `adjust_vibrance_saturation`
6. `crop_and_straighten`
7. `denoise`
8. `sharpen`

### 完成标准

1. 每个第一批工具包都有清晰的参数归一化规则。
2. 参数越界时有稳定裁剪逻辑。
3. 调参结果具备可解释性。

## 12. 阶段 9：先做确定性执行器

### 目标

跑通第一批 8 个工具包在 `whole_image` 模式下的执行链。

### 你要做什么

1. 在 `opencv_tools.py` 定义基础图像调节接口。
2. 在 `image_ops.py` 定义基础图像调节接口。
3. 在执行器中把工具包请求映射到这些工具。

### 先做哪些操作

1. 曝光
2. 高光阴影平衡
3. 对比度
4. 白平衡
5. 自然饱和度 / 饱和度
6. 裁剪与拉直
7. 去噪
8. 锐化

### 文件

1. [app/tools/opencv_tools.py](/Users/liuyan/Desktop/PsAgent/app/tools/opencv_tools.py)
2. [app/tools/image_ops.py](/Users/liuyan/Desktop/PsAgent/app/tools/image_ops.py)
3. [app/graph/subgraphs/deterministic_edit.py](/Users/liuyan/Desktop/PsAgent/app/graph/subgraphs/deterministic_edit.py)

### 完成标准

1. 显式修图最小闭环可以跑通。
2. `candidate_outputs` 有统一格式。

## 13. 阶段 10：先补工具单元测试

### 目标

先验证这 8 个工具包在固定输入下是可控、可重复的。

### 你要做什么

1. 给每个工具包准备最小测试图夹具。
2. 测参数归一化是否落在预期范围。
3. 测输出是否成功生成。
4. 测关键图像指标是否按预期变化。

### 当前阶段建议重点测试

1. 平均亮度变化
2. 对比度变化
3. 色温或色彩变化
4. 输出尺寸是否保持正确
5. 异常参数是否能被裁剪或兜底
6. 需要局部区域时，用真实图片实时请求阿里云分割并验证 mask 内外差异

### 当前测试前置条件

1. 在 `tests/` 下放真实测试图片。
2. 安装 `requirements.txt` 中的依赖。
3. 配置阿里云环境变量：
   `ALIBABA_CLOUD_ACCESS_KEY_ID`
   `ALIBABA_CLOUD_ACCESS_KEY_SECRET`
   可选：
   `ALIBABA_CLOUD_REGION_ID`
   `ALIYUN_IMAGESEG_ENDPOINT`

### 完成标准

1. 第一批 8 个工具包都有基础单元测试。
2. 参数归一化和执行链可以联动验证。
3. 工具层具备最小回归保障。
4. `adjust_exposure` 已经具备“真实图片 + 阿里云实时 mask”测试链路。

## 14. 阶段 11：实现结果评估节点

### 目标

让系统不仅能修，还能判断这次修图结果是否可信。

### 你要做什么

1. 检查有没有输出结果。
2. 检查是否符合计划。
3. 检查是否有明显风险。
4. 决定是否进入人工审核。

### 第一版建议先做规则评估

例如：

1. 计划里有高风险操作则标记审核
2. 结果缺失则标记失败
3. 自动模式下的激进操作则标记审核

### 文件

[app/graph/nodes/evaluate_result.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/evaluate_result.py)

### 完成标准

1. 系统能产出 `eval_report`。
2. 审核分支可以被驱动。

## 15. 阶段 12：实现长期记忆读取

### 目标

让 Planner 在生成计划之前能读到用户长期偏好。

### 你要做什么

1. 定义长期偏好结构。
2. 写最小检索接口。
3. 在 `load_context` 中接入它。

### 第一版建议先做什么

1. 只按 `domain` 过滤
2. 只读 profile
3. 不做复杂语义搜索

### 文件

1. [app/memory/profile.py](/Users/liuyan/Desktop/PsAgent/app/memory/profile.py)
2. [app/memory/retriever.py](/Users/liuyan/Desktop/PsAgent/app/memory/retriever.py)
3. [app/graph/nodes/load_context.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/load_context.py)

### 完成标准

1. `build_plan` 能拿到最小长期偏好输入。

## 16. 阶段 13：实现长期记忆写入

### 目标

让系统开始具备“越用越懂用户”的能力。

### 你要做什么

1. 从本轮交互中抽取稳定偏好。
2. 区分显式偏好和一次性要求。
3. 写入长期记忆。
4. 记录证据次数。

### 第一版建议先支持

1. 显式表达写入
2. 负反馈写入
3. 简单置信度逻辑

### 文件

1. [app/memory/extractor.py](/Users/liuyan/Desktop/PsAgent/app/memory/extractor.py)
2. [app/memory/store.py](/Users/liuyan/Desktop/PsAgent/app/memory/store.py)
3. [app/graph/nodes/update_memory.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/update_memory.py)

### 完成标准

1. 长期偏好能形成最小闭环。

## 17. 阶段 14：先做阿里云分割工具封装

### 目标

为第二批“同一批工具包的局部 `region` 模式”准备统一区域能力。

### 你要做什么

1. 在 `segmentation_tools.py` 中定义统一接口。
2. 区分人像分割与主体分割两类调用。
3. 统一输出 `mask`、`confidence`、`source` 等字段。
4. 做最小错误处理和结果兜底。

### 当前建议接入的阿里云能力

1. `SegmentHDBody`
   用于人像主体分割。
2. `SegmentCommonImage` 或 `SegmentHDCommonImage`
   用于主视觉主体分割。

### 完成标准

1. 人像图可以拿到统一的人像分割结果。
2. 主体图可以拿到统一的主体分割结果。
3. 上层节点不需要知道阿里云具体 API 参数格式。

## 18. 阶段 15：实现 Hybrid 执行器

### 目标

支持同一批工具包在局部 `region` 模式下的“按需分割 -> 局部增强”链路。

### 你要做什么

1. 接入阿里云分割结果。
2. 对需要 mask 的工具包先补区域。
3. 再做局部确定性增强。
4. 统一“区域 + 参数增强”的执行格式。

### 文件

1. [app/tools/segmentation_tools.py](/Users/liuyan/Desktop/PsAgent/app/tools/segmentation_tools.py)
2. [app/graph/subgraphs/hybrid_edit.py](/Users/liuyan/Desktop/PsAgent/app/graph/subgraphs/hybrid_edit.py)

### 完成标准

1. 已有工具包在局部 `region` 模式下可以稳定执行。
2. 输出格式与其他执行器一致。

## 19. 阶段 16：实现生成式执行器

### 目标

让系统具备至少一个复杂改图能力。

### 你要做什么

1. 定义图像编辑模型调用接口。
2. 定义输入图、mask、提示词的组织方式。
3. 在 `execute_generative` 中统一封装调用。

### 当前建议

当前阶段先保留骨架，不进入主开发序列。

如果后面需要恢复这一阶段，再考虑先做一个场景：

1. 去路人
2. 或去杂物

### 文件

1. [app/tools/image_edit_model.py](/Users/liuyan/Desktop/PsAgent/app/tools/image_edit_model.py)
2. [app/graph/subgraphs/generative_edit.py](/Users/liuyan/Desktop/PsAgent/app/graph/subgraphs/generative_edit.py)

### 完成标准

1. 至少一个高价值复杂场景跑通。

## 20. 阶段 17：实现自动修图模式

### 目标

在用户不说“怎么修”的情况下，系统也能给出保守方案。

### 你要做什么

1. 在 `parse_request` 中识别自动模式。
2. 在 `build_plan` 中增加自动模式策略。
3. 根据 domain 给出默认保守方案。
4. 受长期偏好和强度上限约束。

### 建议第一版只做

1. 人像自动修图
2. 主体照片自动修图

### 完成标准

1. 用户只传图时不报错。
2. 系统能产出保守的 `edit_plan`。

## 21. 阶段 18：实现人工审核

### 目标

让系统具备中断确认能力。

### 你要做什么

1. 在 `human_review` 节点接入 `interrupt`
2. 定义审核通过后的恢复路径
3. 定义审核拒绝后的恢复路径

### 第一版建议哪些场景必须审核

1. 自动模式下强度过高的局部增强
2. 涉及人脸区域但结果评估不稳定
3. 后续接入生成式执行器后的高风险内容编辑

### 文件

[app/graph/nodes/human_review.py](/Users/liuyan/Desktop/PsAgent/app/graph/nodes/human_review.py)

### 完成标准

1. 主图能暂停。
2. 主图能恢复。

## 22. 阶段 19：最后再接 API

### 目标

在 Graph 契约稳定后，再把 API 包在外面。

### 你要做什么

1. 设计 `POST /edit`
2. 设计 `POST /feedback`
3. 设计 `POST /resume-review`

### 为什么最后做

1. Graph 变化时不会牵连 API 层反复重构。
2. 先把核心工作流做稳，API 接入会简单很多。

## 23. 每个阶段都要保留的产物

建议每完成一个阶段，都留下下面几种产物：

1. 一份输入状态样例
2. 一份输出状态样例
3. 一份失败案例
4. 一份“还缺什么”的清单

这样后面排查问题会轻松很多。

## 24. 第一版范围控制建议

为了避免项目过早失控，第一版建议明确不做：

1. 太多图片 domain
2. 多图联合修图
3. 复杂风格迁移
4. 开放域实例级内容删除或替换
5. 全自动高风险生成式改图
6. 异步复杂记忆总结

第一版最稳的目标只有五件事：

1. 能理解
2. 能规划
3. 能路由
4. 能执行
5. 能记住最小稳定偏好

做到这五件事，这个项目就已经是一个真正可演进的 Agent 了。
