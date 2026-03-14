package org.bsc.langgraph4j;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import org.bsc.langgraph4j.action.AsyncNodeAction;
import org.bsc.langgraph4j.checkpoint.MemorySaver;
import org.bsc.langgraph4j.langchain4j.serializer.std.ChatMesssageSerializer;
import org.bsc.langgraph4j.langchain4j.serializer.std.ToolExecutionRequestSerializer;
import org.bsc.langgraph4j.prebuilt.MessagesState;
import org.bsc.langgraph4j.serializer.std.ObjectStreamStateSerializer;
import org.bsc.langgraph4j.state.Channel;
import org.bsc.langgraph4j.state.Channels;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Best-Practice: Multi-turn Agent System with Human-in-the-Loop, Tools, Skills and Subagents.
 *
 * <p>This class demonstrates a complete, runnable multi-agent architecture that covers:
 * <ol>
 *   <li><b>Multi-turn conversations</b> – {@link MemorySaver} + {@code threadId} accumulate
 *       the message history across separate {@code graph.stream()} calls.</li>
 *   <li><b>Human-in-the-loop (HITL)</b> – {@link CompileConfig#interruptBefore(String...)}
 *       pauses execution before the {@code human_review} node; the caller resumes by passing
 *       {@link GraphInput#resume(Map)} with the human's decision.</li>
 *   <li><b>Tools</b> – Each skill exposes domain-specific tools via {@code @Tool} annotations
 *       (simulated here for testability; swap in a real {@code ChatModel} for production).</li>
 *   <li><b>Skills</b> – Specialised node actions ({@code research_agent}, {@code coding_agent})
 *       encapsulate a particular capability.</li>
 *   <li><b>Subagents</b> – The {@code calculator_agent} skill is implemented as a fully
 *       independent {@link CompiledGraph} subgraph wired into the parent graph.</li>
 * </ol>
 *
 * <h2>Graph topology</h2>
 * <pre>
 *   START
 *     │
 *   ┌─┴──────────┐
 *   │ supervisor │  (routes based on the last message)
 *   └─┬──────────┘
 *     │
 *     ├──→ research_agent  ──┐
 *     │                      │
 *     ├──→ coding_agent    ──┤
 *     │                      ├──→ supervisor (loop)
 *     ├──→ calculator_agent  ┘   (→ END when no skill matches)
 *     │     (subgraph: parse_input → compute)
 *     │
 *     └──→ human_review  ─── END
 *              ▲
 *        HITL interrupt
 * </pre>
 *
 * <h2>Running against a real LLM</h2>
 * Replace the stub node actions with an {@code AgentExecutor} backed by OpenAI/Ollama:
 * <pre>{@code
 * var agentGraph = AgentExecutor.builder()
 *         .chatModel(OpenAiChatModel.builder().apiKey(System.getenv("OPENAI_API_KEY")).build())
 *         .toolsFromObject(new ResearchTools())
 *         .build();
 * // Then add it as a subgraph:
 * graph.addNode("research_agent", agentGraph.compile());
 * }</pre>
 */
public class MultiTurnAgentSystemTest {

    private static final org.slf4j.Logger log =
            org.slf4j.LoggerFactory.getLogger(MultiTurnAgentSystemTest.class);

    // =========================================================================
    // 1. STATE DEFINITION
    // =========================================================================

    /**
     * Shared state for the entire agent system.
     *
     * <ul>
     *   <li>{@code messages}      – appender channel; every node <em>appends</em> new
     *       {@link ChatMessage}s, giving all downstream nodes the full history.</li>
     *   <li>{@code next}          – routing hint written by {@code human_review} to signal
     *       whether execution should {@code "continue"} or {@code "stop"}.</li>
     *   <li>{@code human_approved} – Boolean decision provided by the human operator.</li>
     *   <li>{@code human_feedback} – Free-text explanation from the human operator.</li>
     *   <li>{@code skill_result}  – Intermediate result written by the active skill node.</li>
     * </ul>
     */
    static class AgentSystemState extends MessagesState<ChatMessage> {

        /**
         * Schema: inherit the {@code messages} appender channel from {@link MessagesState}.
         * All other keys use the default replace-on-update semantics.
         */
        static final Map<String, Channel<?>> SCHEMA = Map.of(
                MESSAGES_STATE, Channels.appender(ArrayList::new)
        );

        AgentSystemState(Map<String, Object> initData) {
            super(initData);
        }

        /** Routing hint used after {@code human_review}. */
        Optional<String> next() {
            return value("next");
        }

        /** Human's approval decision (present only after HITL resume). */
        Optional<Boolean> humanApproved() {
            return value("human_approved");
        }

        /** Human's free-text feedback (present only after HITL resume). */
        Optional<String> humanFeedback() {
            return value("human_feedback");
        }

        /** Result written by the last executed skill node. */
        Optional<String> skillResult() {
            return value("skill_result");
        }
    }

    // =========================================================================
    // 2. SERIALIZER
    // =========================================================================

    /**
     * State serializer that knows how to handle LangChain4j message types.
     * Required for checkpoint persistence ({@link MemorySaver} and DB-backed savers).
     */
    static class AgentSystemStateSerializer
            extends ObjectStreamStateSerializer<AgentSystemState> {

        AgentSystemStateSerializer() {
            super(AgentSystemState::new);
            // Register LangChain4j type adapters
            mapper().register(ToolExecutionRequest.class, new ToolExecutionRequestSerializer());
            mapper().register(ChatMessage.class, new ChatMesssageSerializer());
        }
    }

    // =========================================================================
    // 3. SUPERVISOR – keyword-based routing
    // =========================================================================

    /**
     * Determines which skill to invoke next based on the content of the last message.
     *
     * <p>In production replace keyword matching with an LLM call that returns a structured
     * routing decision (see {@code MultiAgentSupervisorITest} for an Ollama/OpenAI example).
     */
    static String supervisorRoute(AgentSystemState state) {
        var lastMsg = state.lastMessage().orElse(null);
        if (lastMsg == null) {
            return END;
        }

        // Only dispatch to a skill when the latest message comes from the user.
        // After a skill node appends its AI response the supervisor simply ends
        // the current turn; the next turn starts when the user sends a new message.
        if (lastMsg.type() != dev.langchain4j.data.message.ChatMessageType.USER) {
            return END;
        }

        var text = ((UserMessage) lastMsg).singleText().toLowerCase(Locale.ROOT);

        // Check coding BEFORE research to avoid "write code for a binary search"
        // routing to research because the word "search" appears in the sentence.
        if (text.contains("code") || text.contains("program")
                || text.contains("implement") || text.contains("write")) {
            return "coding_agent";
        }
        if (text.contains("search") || text.contains("research")
                || text.contains("find")   || text.contains("what is")) {
            return "research_agent";
        }
        if (text.contains("calculate") || text.contains("math")
                || text.contains("compute")  || text.contains("sum")) {
            return "calculator_agent";
        }
        if (text.contains("sensitive") || text.contains("delete")
                || text.contains("deploy")   || text.contains("dangerous")) {
            return "human_review";
        }

        // No recognised keyword → end the current turn
        return END;
    }

    // =========================================================================
    // 4. SKILLS (node actions)
    // =========================================================================

    /**
     * <b>Research skill</b> – simulates a web-search agent.
     *
     * <p>In production wire a real LLM + search tool here, e.g.:
     * <pre>{@code
     * AgentExecutor.builder()
     *     .chatModel(model)
     *     .toolsFromObject(new ResearchTools())
     *     .build()
     *     .compile()
     * }</pre>
     */
    static AsyncNodeAction<AgentSystemState> researchAgent() {
        return state -> CompletableFuture.supplyAsync(() -> {
            var query = state.lastMessage()
                    .filter(m -> m.type() == dev.langchain4j.data.message.ChatMessageType.USER)
                    .map(m -> ((UserMessage) m).singleText())
                    .orElse("unknown query");

            log.info("[research_agent] query='{}'", query);

            var result = "Research results for «%s»: key findings – A, B, C.".formatted(query);

            return Map.<String, Object>of(
                    "messages",      AiMessage.from(result),
                    "skill_result",  result
            );
        });
    }

    /**
     * <b>Coding skill</b> – simulates a code-generation agent.
     *
     * <p>In production wire an {@code AgentExecutor} with coding tools.
     */
    static AsyncNodeAction<AgentSystemState> codingAgent() {
        return state -> CompletableFuture.supplyAsync(() -> {
            var request = state.lastMessage()
                    .filter(m -> m.type() == dev.langchain4j.data.message.ChatMessageType.USER)
                    .map(m -> ((UserMessage) m).singleText())
                    .orElse("unknown request");

            log.info("[coding_agent] request='{}'", request);

            var result = """
                    Here is the Java implementation for: %s
                    ```java
                    public class Solution {
                        public static void main(String[] args) {
                            // TODO: implement
                        }
                    }
                    ```""".formatted(request);

            return Map.<String, Object>of(
                    "messages",      AiMessage.from(result),
                    "skill_result",  result
            );
        });
    }

    // =========================================================================
    // 5. CALCULATOR SUBAGENT (compiled subgraph)
    // =========================================================================

    /**
     * Builds the <b>calculator subgraph</b>, which is an independent {@link StateGraph}
     * compiled and injected as a node into the parent graph.
     *
     * <p>This demonstrates the <em>subagent</em> pattern: each skill can be a self-contained
     * graph with its own nodes, edges, and (optionally) its own checkpoint saver.
     *
     * <pre>
     *   START → parse_input → compute → END
     * </pre>
     */
    static CompiledGraph<AgentSystemState> calculatorSubgraph(
            AgentSystemStateSerializer serializer) throws GraphStateException {

        return new StateGraph<>(AgentSystemState.SCHEMA, serializer)
                // Node 1: extract the expression from the last user message
                .addNode("parse_input", node_async(state -> {
                    var lastMsg = state.lastMessage()
                            .filter(m -> m.type() == dev.langchain4j.data.message.ChatMessageType.USER)
                            .map(m -> ((UserMessage) m).singleText())
                            .orElse("0");
                    log.info("[calculator/parse_input] expression='{}'", lastMsg);
                    return Map.of("skill_result", "parsed: " + lastMsg);
                }))
                // Node 2: perform the computation (simplified stub)
                .addNode("compute", node_async(state -> {
                    var expr = state.skillResult().orElse("?");
                    log.info("[calculator/compute] evaluating '{}'", expr);
                    var answer = "Result = 42  (expression: " + expr + ")";
                    return Map.of(
                            "messages",     AiMessage.from(answer),
                            "skill_result", answer
                    );
                }))
                .addEdge(START, "parse_input")
                .addEdge("parse_input", "compute")
                .addEdge("compute", END)
                .compile();
    }

    // =========================================================================
    // 6. HUMAN REVIEW NODE
    // =========================================================================

    /**
     * <b>Human-review node</b> – processes the human operator's decision.
     *
     * <p>Execution reaches this node <em>after</em> the human has resumed the graph with
     * {@link GraphInput#resume(Map)} containing {@code human_approved} and
     * {@code human_feedback}. The node records the decision as an {@link AiMessage} and
     * sets the {@code next} routing key.
     */
    static AsyncNodeAction<AgentSystemState> humanReviewNode() {
        return state -> CompletableFuture.supplyAsync(() -> {
            boolean approved = state.humanApproved().orElse(false);
            String  feedback = state.humanFeedback().orElse("(no feedback)");

            log.info("[human_review] approved={} feedback='{}'", approved, feedback);

            String response = approved
                    ? "✅ Human approved. Proceeding. Feedback: " + feedback
                    : "❌ Human rejected. Aborting.  Feedback: " + feedback;

            return Map.<String, Object>of(
                    "messages",      AiMessage.from(response),
                    "skill_result",  response,
                    "next",          approved ? "continue" : "stop"
            );
        });
    }

    // =========================================================================
    // 7. GRAPH BUILDER
    // =========================================================================

    /**
     * Assembles the complete multi-turn agent graph.
     *
     * @param saver optional checkpoint saver; pass a {@link MemorySaver} to enable
     *              multi-turn conversations and HITL persistence.
     * @return compiled {@link CompiledGraph} ready to stream inputs
     */
    CompiledGraph<AgentSystemState> buildGraph(MemorySaver saver) throws GraphStateException {
        var serializer    = new AgentSystemStateSerializer();
        var calcSubgraph  = calculatorSubgraph(serializer);

        var stateGraph = new StateGraph<>(AgentSystemState.SCHEMA, serializer)

                // ── NODES ──────────────────────────────────────────────────

                // supervisor: passthrough – routing logic lives in the conditional edge
                .addNode("supervisor",        node_async(state -> Map.of()))

                // skills
                .addNode("research_agent",   researchAgent())
                .addNode("coding_agent",      codingAgent())
                .addNode("calculator_agent",  calcSubgraph)   // ← subgraph node

                // human-in-the-loop
                .addNode("human_review",      humanReviewNode())

                // ── EDGES ──────────────────────────────────────────────────

                .addEdge(START, "supervisor")

                // supervisor → skill (or END)
                .addConditionalEdges(
                        "supervisor",
                        edge_async(MultiTurnAgentSystemTest::supervisorRoute),
                        Map.of(
                                "research_agent",   "research_agent",
                                "coding_agent",     "coding_agent",
                                "calculator_agent", "calculator_agent",
                                "human_review",     "human_review",
                                END,                END
                        )
                )

                // skill agents loop back to supervisor (supervisor will then route to END
                // when the last message is an AI response with no routing keyword)
                .addEdge("research_agent",   "supervisor")
                .addEdge("coding_agent",     "supervisor")
                .addEdge("calculator_agent", "supervisor")

                // human_review exits based on human decision
                .addConditionalEdges(
                        "human_review",
                        edge_async(state -> state.next().orElse(END)),
                        Map.of(
                                "continue", END,   // approved  → normal end
                                "stop",     END,   // rejected  → also end
                                END,        END
                        )
                );

        var compileConfigBuilder = CompileConfig.builder();
        if (saver != null) {
            compileConfigBuilder.checkpointSaver(saver);
        }
        return stateGraph.compile(compileConfigBuilder.build());
    }

    // =========================================================================
    // 8. TESTS
    // =========================================================================

    // ── Helper ──────────────────────────────────────────────────────────────

    /**
     * Collects all {@link NodeOutput} events from a single graph run into a list.
     */
    private List<NodeOutput<AgentSystemState>> run(
            CompiledGraph<AgentSystemState> graph,
            Map<String, Object> inputs,
            RunnableConfig config) {
        return graph.stream(inputs, config)
                .stream()
                .peek(e -> log.info("  NodeOutput: node={}", e.node()))
                .collect(Collectors.toList());
    }

    /**
     * Resumes a previously interrupted graph with data provided by the human operator.
     */
    private List<NodeOutput<AgentSystemState>> resume(
            CompiledGraph<AgentSystemState> graph,
            Map<String, Object> humanData,
            RunnableConfig config) {
        return graph.stream(GraphInput.resume(humanData), config)
                .stream()
                .peek(e -> log.info("  NodeOutput (resume): node={}", e.node()))
                .collect(Collectors.toList());
    }

    // ── Skill tests ──────────────────────────────────────────────────────────

    @Test
    void testResearchSkill() throws Exception {
        var graph = buildGraph(null);
        var cfg   = RunnableConfig.builder().build();

        var outputs = run(graph, Map.of("messages", UserMessage.from("search for AI trends")), cfg);

        // Expected flow: START → supervisor → research_agent → supervisor → END
        var nodeNames = outputs.stream().map(NodeOutput::node).collect(Collectors.toList());
        assertTrue(nodeNames.contains("research_agent"),
                "research_agent must execute; nodes were: " + nodeNames);

        var finalState = outputs.get(outputs.size() - 1).state();
        assertTrue(finalState.skillResult().isPresent(), "skill_result must be set");
        assertTrue(finalState.skillResult().get().contains("Research results"),
                "skill_result must contain research output");

        log.info("Research skill result: {}", finalState.skillResult().get());
    }

    @Test
    void testCodingSkill() throws Exception {
        var graph = buildGraph(null);
        var cfg   = RunnableConfig.builder().build();

        var outputs = run(graph,
                Map.of("messages", UserMessage.from("write code for a binary search")), cfg);

        var nodeNames = outputs.stream().map(NodeOutput::node).collect(Collectors.toList());
        assertTrue(nodeNames.contains("coding_agent"),
                "coding_agent must execute; nodes were: " + nodeNames);

        var finalState = outputs.get(outputs.size() - 1).state();
        assertTrue(finalState.skillResult().isPresent());
        assertTrue(finalState.skillResult().get().contains("Java implementation"),
                "skill_result must contain generated code");

        log.info("Coding skill result:\n{}", finalState.skillResult().get());
    }

    @Test
    void testCalculatorSubagent() throws Exception {
        var graph = buildGraph(null);
        var cfg   = RunnableConfig.builder().build();

        var outputs = run(graph,
                Map.of("messages", UserMessage.from("calculate the sum of 1 to 100")), cfg);

        // Expected flow: supervisor → calculator_agent (subgraph: parse_input → compute) → supervisor → END
        var nodeNames = outputs.stream().map(NodeOutput::node).collect(Collectors.toList());
        assertTrue(nodeNames.contains("calculator_agent"),
                "calculator_agent (subgraph) must execute; nodes were: " + nodeNames);

        var finalState = outputs.get(outputs.size() - 1).state();
        assertTrue(finalState.skillResult().isPresent());
        assertTrue(finalState.skillResult().get().contains("Result"),
                "skill_result must contain computation result");

        log.info("Calculator subagent result: {}", finalState.skillResult().get());
    }

    // ── Multi-turn conversation test ─────────────────────────────────────────

    /**
     * Verifies that messages accumulate across multiple turns thanks to the
     * {@link MemorySaver} and a shared {@code threadId}.
     *
     * <pre>
     *   Turn 1: "search for quantum computing"
     *   Turn 2: "write code for a quantum circuit"
     *   Turn 3: "calculate pi to 5 decimal places"
     * </pre>
     *
     * After turn 3 the {@code messages} list must contain contributions from all three turns.
     */
    @Test
    void testMultiTurnConversation() throws Exception {
        var saver = new MemorySaver();
        var graph = buildGraph(saver);

        // Each turn uses the same config (same threadId)
        var cfg = RunnableConfig.builder()
                .threadId("test-session-42")
                .build();

        // ── Turn 1: research ────────────────────────────────────────────────
        log.info("=== Turn 1: research ===");
        var outputs1 = run(graph,
                Map.of("messages", UserMessage.from("search for quantum computing")), cfg);
        var state1 = outputs1.get(outputs1.size() - 1).state();
        assertTrue(state1.messages().size() >= 2, "Turn 1 must have at least user + AI message");
        assertTrue(outputs1.stream().anyMatch(o -> "research_agent".equals(o.node())),
                "research_agent must have executed in turn 1");
        log.info("Turn 1 message count: {}", state1.messages().size());

        // ── Turn 2: coding ──────────────────────────────────────────────────
        log.info("=== Turn 2: coding ===");
        var outputs2 = run(graph,
                Map.of("messages", UserMessage.from("write code for a quantum circuit")), cfg);
        var state2 = outputs2.get(outputs2.size() - 1).state();
        // Messages from turn 1 are still there (appender channel + checkpoint merge)
        assertTrue(state2.messages().size() > state1.messages().size(),
                "Turn 2 must have MORE messages than turn 1 (conversation history accumulates)");
        assertTrue(outputs2.stream().anyMatch(o -> "coding_agent".equals(o.node())),
                "coding_agent must have executed in turn 2");
        log.info("Turn 2 message count: {}", state2.messages().size());

        // ── Turn 3: calculator ──────────────────────────────────────────────
        log.info("=== Turn 3: calculator ===");
        var outputs3 = run(graph,
                Map.of("messages", UserMessage.from("calculate pi to 5 decimal places")), cfg);
        var state3 = outputs3.get(outputs3.size() - 1).state();
        assertTrue(state3.messages().size() > state2.messages().size(),
                "Turn 3 must have MORE messages than turn 2");
        assertTrue(outputs3.stream().anyMatch(o -> "calculator_agent".equals(o.node())),
                "calculator_agent must have executed in turn 3");
        log.info("Turn 3 final message count: {}", state3.messages().size());

        // Verify full conversation history is preserved
        var allText = state3.messages().stream()
                .map(m -> switch (m.type()) {
                    case USER -> ((UserMessage) m).singleText();
                    case AI   -> ((AiMessage) m).text();
                    default   -> "";
                })
                .collect(Collectors.joining(" | "));
        log.info("Full conversation:\n{}", allText);

        assertTrue(allText.contains("quantum computing"),  "History must contain turn-1 query");
        assertTrue(allText.contains("quantum circuit"),   "History must contain turn-2 query");
        assertTrue(allText.contains("pi"),               "History must contain turn-3 query");
    }

    // ── Human-in-the-loop tests ──────────────────────────────────────────────

    /**
     * Tests the <em>approve</em> path of the HITL flow.
     *
     * <pre>
     *   1. Send a "sensitive delete" message
     *      → supervisor routes to human_review
     *      → graph is INTERRUPTED before human_review executes
     *   2. Human approves
     *      → graph resumes, human_review executes with approved=true
     *      → human_review → END
     * </pre>
     */
    @Test
    void testHumanInTheLoop_approve() throws Exception {
        var saver = new MemorySaver();
        var graph = buildGraph(saver);

        // Compile with interruptBefore("human_review")
        var compileConfig = CompileConfig.builder()
                .checkpointSaver(saver)
                .interruptBefore("human_review")
                .build();
        // Recompile with interrupt config (buildGraph uses its own compile step above)
        var serializer   = new AgentSystemStateSerializer();
        var calcSubgraph = calculatorSubgraph(serializer);
        var graphWithHitl = new StateGraph<>(AgentSystemState.SCHEMA, serializer)
                .addNode("supervisor",        node_async(state -> Map.of()))
                .addNode("research_agent",    researchAgent())
                .addNode("coding_agent",      codingAgent())
                .addNode("calculator_agent",  calcSubgraph)
                .addNode("human_review",      humanReviewNode())
                .addEdge(START, "supervisor")
                .addConditionalEdges("supervisor",
                        edge_async(MultiTurnAgentSystemTest::supervisorRoute),
                        Map.of("research_agent", "research_agent",
                                "coding_agent", "coding_agent",
                                "calculator_agent", "calculator_agent",
                                "human_review", "human_review",
                                END, END))
                .addEdge("research_agent",   "supervisor")
                .addEdge("coding_agent",     "supervisor")
                .addEdge("calculator_agent", "supervisor")
                .addConditionalEdges("human_review",
                        edge_async(state -> state.next().orElse(END)),
                        Map.of("continue", END, "stop", END, END, END))
                .compile(compileConfig);

        var cfg = RunnableConfig.builder().threadId("hitl-thread-1").build();

        // ── Step 1: Trigger the sensitive action ────────────────────────────
        log.info("=== HITL step 1: trigger sensitive action ===");
        var step1 = run(graphWithHitl,
                Map.of("messages", UserMessage.from("delete sensitive production data")), cfg);

        // The graph must have stopped BEFORE human_review
        var nodesStep1 = step1.stream().map(NodeOutput::node).collect(Collectors.toList());
        log.info("Nodes executed in step 1: {}", nodesStep1);
        assertFalse(nodesStep1.contains("human_review"),
                "human_review must NOT have executed yet (interrupted before it)");
        assertTrue(nodesStep1.contains("supervisor"),
                "supervisor must have executed to route the message");

        // Verify state is suspended at supervisor output
        var suspendedState = step1.get(step1.size() - 1).state();
        assertFalse(suspendedState.humanApproved().isPresent(),
                "human_approved must not be set before human acts");

        // ── Step 2: Human approves ──────────────────────────────────────────
        log.info("=== HITL step 2: human approves ===");
        var step2 = resume(graphWithHitl,
                Map.of(
                        "human_approved", true,
                        "human_feedback", "Approved by admin – proceed with caution"
                ), cfg);

        var nodesStep2 = step2.stream().map(NodeOutput::node).collect(Collectors.toList());
        log.info("Nodes executed in step 2 (after resume): {}", nodesStep2);
        assertTrue(nodesStep2.contains("human_review"),
                "human_review MUST have executed after resume");

        var finalState = step2.get(step2.size() - 1).state();
        assertTrue(finalState.humanApproved().orElse(false), "human_approved must be true");
        assertTrue(finalState.skillResult().isPresent());
        assertTrue(finalState.skillResult().get().contains("approved"),
                "skill_result should mention approval");
        log.info("HITL approved result: {}", finalState.skillResult().get());
    }

    /**
     * Tests the <em>reject</em> path of the HITL flow.
     */
    @Test
    void testHumanInTheLoop_reject() throws Exception {
        var saver     = new MemorySaver();
        var serializer   = new AgentSystemStateSerializer();
        var calcSubgraph = calculatorSubgraph(serializer);

        var compileConfig = CompileConfig.builder()
                .checkpointSaver(saver)
                .interruptBefore("human_review")
                .build();

        var graphWithHitl = new StateGraph<>(AgentSystemState.SCHEMA, serializer)
                .addNode("supervisor",        node_async(state -> Map.of()))
                .addNode("research_agent",    researchAgent())
                .addNode("coding_agent",      codingAgent())
                .addNode("calculator_agent",  calcSubgraph)
                .addNode("human_review",      humanReviewNode())
                .addEdge(START, "supervisor")
                .addConditionalEdges("supervisor",
                        edge_async(MultiTurnAgentSystemTest::supervisorRoute),
                        Map.of("research_agent", "research_agent",
                                "coding_agent", "coding_agent",
                                "calculator_agent", "calculator_agent",
                                "human_review", "human_review",
                                END, END))
                .addEdge("research_agent",   "supervisor")
                .addEdge("coding_agent",     "supervisor")
                .addEdge("calculator_agent", "supervisor")
                .addConditionalEdges("human_review",
                        edge_async(state -> state.next().orElse(END)),
                        Map.of("continue", END, "stop", END, END, END))
                .compile(compileConfig);

        var cfg = RunnableConfig.builder().threadId("hitl-thread-reject").build();

        // Step 1: trigger
        run(graphWithHitl,
                Map.of("messages", UserMessage.from("deploy dangerous script to production")), cfg);

        // Step 2: human rejects
        log.info("=== HITL step 2: human rejects ===");
        var step2 = resume(graphWithHitl,
                Map.of(
                        "human_approved", false,
                        "human_feedback", "Rejected – security policy violation"
                ), cfg);

        var nodesStep2 = step2.stream().map(NodeOutput::node).collect(Collectors.toList());
        assertTrue(nodesStep2.contains("human_review"), "human_review must execute after resume");

        var finalState = step2.get(step2.size() - 1).state();
        assertFalse(finalState.humanApproved().orElse(true), "human_approved must be false");
        assertTrue(finalState.skillResult().isPresent());
        assertTrue(finalState.skillResult().get().contains("rejected"),
                "skill_result should mention rejection");
        log.info("HITL rejected result: {}", finalState.skillResult().get());
    }

    // ── Full integration flow ────────────────────────────────────────────────

    /**
     * Runs a complete multi-turn scenario that exercises all skills and ends with a HITL
     * approval in the final turn.
     *
     * <pre>
     *   Turn 1: research query  → research_agent
     *   Turn 2: coding request  → coding_agent
     *   Turn 3: math question   → calculator_agent (subgraph)
     *   Turn 4: sensitive op    → INTERRUPTED (human_review) → resumed with approval
     * </pre>
     */
    @Test
    void testFullWorkflow() throws Exception {
        var saver = new MemorySaver();
        var serializer   = new AgentSystemStateSerializer();
        var calcSubgraph = calculatorSubgraph(serializer);

        // Build graph WITH checkpointing and interrupt
        var compileConfig = CompileConfig.builder()
                .checkpointSaver(saver)
                .interruptBefore("human_review")
                .build();

        var stateGraph = new StateGraph<>(AgentSystemState.SCHEMA, serializer)
                .addNode("supervisor",        node_async(state -> Map.of()))
                .addNode("research_agent",    researchAgent())
                .addNode("coding_agent",      codingAgent())
                .addNode("calculator_agent",  calcSubgraph)
                .addNode("human_review",      humanReviewNode())
                .addEdge(START, "supervisor")
                .addConditionalEdges("supervisor",
                        edge_async(MultiTurnAgentSystemTest::supervisorRoute),
                        Map.of("research_agent", "research_agent",
                                "coding_agent", "coding_agent",
                                "calculator_agent", "calculator_agent",
                                "human_review", "human_review",
                                END, END))
                .addEdge("research_agent",   "supervisor")
                .addEdge("coding_agent",     "supervisor")
                .addEdge("calculator_agent", "supervisor")
                .addConditionalEdges("human_review",
                        edge_async(state -> state.next().orElse(END)),
                        Map.of("continue", END, "stop", END, END, END));

        var graph = stateGraph.compile(compileConfig);
        var cfg   = RunnableConfig.builder().threadId("full-workflow-thread").build();

        // Turn 1
        log.info("=== Full workflow Turn 1: research ===");
        var t1 = run(graph, Map.of("messages", UserMessage.from("search for machine learning trends")), cfg);
        assertTrue(t1.stream().anyMatch(o -> "research_agent".equals(o.node())));

        // Turn 2
        log.info("=== Full workflow Turn 2: coding ===");
        var t2 = run(graph, Map.of("messages", UserMessage.from("write code for a neural network")), cfg);
        assertTrue(t2.stream().anyMatch(o -> "coding_agent".equals(o.node())));

        // Turn 3
        log.info("=== Full workflow Turn 3: calculator (subagent) ===");
        var t3 = run(graph, Map.of("messages", UserMessage.from("calculate the Fibonacci sum")), cfg);
        assertTrue(t3.stream().anyMatch(o -> "calculator_agent".equals(o.node())));

        // Turn 4a: trigger sensitive action – graph is interrupted
        log.info("=== Full workflow Turn 4a: trigger HITL ===");
        var t4a = run(graph, Map.of("messages", UserMessage.from("delete all training data")), cfg);
        var t4aNodes = t4a.stream().map(NodeOutput::node).collect(Collectors.toList());
        assertFalse(t4aNodes.contains("human_review"),
                "Graph must be interrupted before human_review; nodes=" + t4aNodes);

        // Turn 4b: human approves
        log.info("=== Full workflow Turn 4b: human approves ===");
        var t4b = resume(graph, Map.of("human_approved", true, "human_feedback", "OK"), cfg);
        assertTrue(t4b.stream().anyMatch(o -> "human_review".equals(o.node())),
                "human_review must execute after HITL resume");

        var finalState = t4b.get(t4b.size() - 1).state();
        log.info("Full workflow final message count: {}", finalState.messages().size());
        // Full history: turn-1 msgs + turn-2 msgs + turn-3 msgs + turn-4 msgs
        assertTrue(finalState.messages().size() >= 8,
                "Full history must accumulate across all 4 turns");
    }
}
