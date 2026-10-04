#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <nlohmann/json.hpp>

// Model Context Protocol server (Streamable HTTP transport, JSON responses only) on 127.0.0.1.
// A background thread accepts connections one at a time and answers the protocol itself; tool calls are
// queued and run on the main thread by poll(), so they can touch the scene.
class McpServer {
public:
    struct ToolResult {
        std::string text;
        bool isError = false;
    };
    using ToolHandler = std::function<ToolResult(const std::string& name, const nlohmann::json& arguments)>;

    // tools: the MCP tool descriptions ({name, description, inputSchema}); fixed for the server's lifetime.
    McpServer(uint16_t port, nlohmann::json tools, std::string instructions);
    ~McpServer();

    McpServer(const McpServer&) = delete;
    McpServer& operator=(const McpServer&) = delete;

    // False when the port could not be bound; error() says why.
    bool running() const { return m_running; }
    const std::string& error() const { return m_error; }
    uint16_t port() const { return m_port; }

    bool hasPendingCalls();
    // Runs the queued tool calls through the handler. Main thread.
    void poll(const ToolHandler& handler);

private:
    enum class CallState { Pending, Running, Done, Cancelled };
    struct PendingCall {
        std::string name;
        nlohmann::json arguments;
        CallState state = CallState::Pending;
        ToolResult result;
    };

    void serve();
    void handleConnection(uintptr_t client);
    // JSON-RPC message -> response; null for notifications and responses.
    nlohmann::json handleMessage(const nlohmann::json& message);
    ToolResult callTool(const std::string& name, const nlohmann::json& arguments);

    uint16_t m_port;
    nlohmann::json m_tools;
    std::string m_instructions;
    std::string m_error;
    bool m_running = false;
    bool m_winsockStarted = false;
    std::atomic<bool> m_stop{ false };
    uintptr_t m_listenSocket; // SOCKET
    std::thread m_thread;

    std::mutex m_mutex;
    std::condition_variable m_callDone;
    std::deque<std::shared_ptr<PendingCall>> m_queue;
};
