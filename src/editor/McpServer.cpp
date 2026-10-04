#include "McpServer.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <map>

using nlohmann::json;

namespace {
constexpr const char* kServerName = "miras-engine";
// Newest first; a client asking for another version gets the newest and decides whether it can use it.
constexpr const char* kProtocolVersions[] = { "2025-11-25", "2025-06-18", "2025-03-26" };
// How long a call waits for the editor to pick it up; it can't while playing or mid-drag.
constexpr auto kPickupTimeout = std::chrono::seconds(30);
constexpr size_t kMaxHeaderSize = 64 * 1024;
constexpr size_t kMaxBodySize = 16 * 1024 * 1024;
constexpr DWORD kSocketTimeoutMs = 5000;

std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string trim(const std::string& s)
{
    const size_t begin = s.find_first_not_of(" \t");
    if (begin == std::string::npos)
        return {};
    return s.substr(begin, s.find_last_not_of(" \t") - begin + 1);
}

bool sendAll(SOCKET s, const std::string& data)
{
    size_t sent = 0;
    while (sent < data.size()) {
        const int n = send(s, data.data() + sent, static_cast<int>(std::min<size_t>(data.size() - sent, 1 << 20)), 0);
        if (n <= 0)
            return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

void sendResponse(SOCKET s, int status, const char* reason, const std::string& body = {}, const char* extraHeaders = "")
{
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\n";
    if (!body.empty())
        head += "Content-Type: application/json\r\n";
    head += "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n";
    head += extraHeaders;
    head += "\r\n";
    sendAll(s, head + body);
}

// "localhost" or "127.0.0.1", optionally with a port. Checked against Host and Origin so a web page
// can't reach the editor through DNS rebinding.
bool isLocalHost(const std::string& host)
{
    for (const char* name : { "localhost", "127.0.0.1", "[::1]" }) {
        const std::string n = name;
        if (host == n || (host.rfind(n + ":", 0) == 0))
            return true;
    }
    return false;
}

bool allowedOrigin(const std::string& origin)
{
    if (origin.empty())
        return true; // not a browser
    for (const char* scheme : { "http://", "https://" }) {
        const std::string s = scheme;
        if (origin.rfind(s, 0) == 0 && isLocalHost(origin.substr(s.size())))
            return true;
    }
    return false;
}

json rpcResult(const json& id, json result)
{
    return { { "jsonrpc", "2.0" }, { "id", id }, { "result", std::move(result) } };
}

json rpcError(const json& id, int code, const std::string& message)
{
    return { { "jsonrpc", "2.0" }, { "id", id }, { "error", { { "code", code }, { "message", message } } } };
}
} // namespace

McpServer::McpServer(uint16_t port, json tools, std::string instructions)
    : m_port(port)
    , m_tools(std::move(tools))
    , m_instructions(std::move(instructions))
    , m_listenSocket(INVALID_SOCKET)
{
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        m_error = "WSAStartup failed";
        return;
    }
    m_winsockStarted = true;

    const SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        m_error = "socket() failed: " + std::to_string(WSAGetLastError());
        return;
    }
    // Keeps another program from binding the same port while we listen on it.
    const BOOL exclusive = TRUE;
    setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(s, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR ||
        listen(s, SOMAXCONN) == SOCKET_ERROR) {
        m_error = "cannot listen on 127.0.0.1:" + std::to_string(port) + " (error " +
            std::to_string(WSAGetLastError()) + "; is another editor running?)";
        closesocket(s);
        return;
    }
    m_listenSocket = s;
    m_running = true;
    m_thread = std::thread(&McpServer::serve, this);
}

McpServer::~McpServer()
{
    {
        std::lock_guard lock(m_mutex);
        m_stop = true;
    }
    m_callDone.notify_all();
    // Closing the socket makes the blocked accept() return.
    if (m_listenSocket != INVALID_SOCKET)
        closesocket(static_cast<SOCKET>(m_listenSocket));
    if (m_thread.joinable())
        m_thread.join();
    if (m_winsockStarted)
        WSACleanup();
}

bool McpServer::hasPendingCalls()
{
    std::lock_guard lock(m_mutex);
    return !m_queue.empty();
}

void McpServer::poll(const ToolHandler& handler)
{
    for (;;) {
        std::shared_ptr<PendingCall> call;
        {
            std::lock_guard lock(m_mutex);
            if (m_queue.empty())
                return;
            call = std::move(m_queue.front());
            m_queue.pop_front();
            call->state = CallState::Running;
        }
        ToolResult result;
        try {
            result = handler(call->name, call->arguments);
        }
        catch (const std::exception& e) {
            result = { e.what(), true };
        }
        {
            std::lock_guard lock(m_mutex);
            call->result = std::move(result);
            call->state = CallState::Done;
        }
        m_callDone.notify_all();
    }
}

McpServer::ToolResult McpServer::callTool(const std::string& name, const json& arguments)
{
    auto call = std::make_shared<PendingCall>();
    call->name = name;
    call->arguments = arguments;
    std::unique_lock lock(m_mutex);
    m_queue.push_back(call);
    const auto deadline = std::chrono::steady_clock::now() + kPickupTimeout;
    m_callDone.wait_until(lock, deadline, [&] { return call->state != CallState::Pending || m_stop; });
    if (call->state == CallState::Pending) {
        call->state = CallState::Cancelled;
        std::erase(m_queue, call);
        if (m_stop)
            return { "The editor is closing", true };
        return { "The editor did not pick up the call. It runs commands only in edit mode (not while playing) "
                 "and waits for mouse drags and text input to finish.", true };
    }
    // Once started, the call is waited for however long it takes.
    m_callDone.wait(lock, [&] { return call->state == CallState::Done; });
    return call->result;
}

void McpServer::serve()
{
    const SOCKET listenSocket = static_cast<SOCKET>(m_listenSocket);
    while (!m_stop) {
        const SOCKET client = accept(listenSocket, nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            if (m_stop)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&kSocketTimeoutMs), sizeof(kSocketTimeoutMs));
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&kSocketTimeoutMs), sizeof(kSocketTimeoutMs));
        handleConnection(static_cast<uintptr_t>(client));
        closesocket(client);
    }
}

void McpServer::handleConnection(uintptr_t clientHandle)
{
    const SOCKET client = static_cast<SOCKET>(clientHandle);
    std::string data;
    char buffer[8192];
    size_t headerEnd;
    while ((headerEnd = data.find("\r\n\r\n")) == std::string::npos) {
        if (data.size() > kMaxHeaderSize) {
            sendResponse(client, 431, "Request Header Fields Too Large");
            return;
        }
        const int n = recv(client, buffer, sizeof(buffer), 0);
        if (n <= 0)
            return;
        data.append(buffer, static_cast<size_t>(n));
    }

    // Request line and headers (names lowercased).
    std::map<std::string, std::string> headers;
    std::string method, target;
    size_t lineStart = 0;
    bool firstLine = true;
    while (lineStart < headerEnd) {
        size_t lineEnd = data.find("\r\n", lineStart);
        if (lineEnd == std::string::npos || lineEnd > headerEnd)
            lineEnd = headerEnd;
        const std::string line = data.substr(lineStart, lineEnd - lineStart);
        lineStart = lineEnd + 2;
        if (firstLine) {
            firstLine = false;
            const size_t a = line.find(' ');
            const size_t b = line.find(' ', a + 1);
            if (a == std::string::npos || b == std::string::npos) {
                sendResponse(client, 400, "Bad Request");
                return;
            }
            method = line.substr(0, a);
            target = line.substr(a + 1, b - a - 1);
            continue;
        }
        const size_t colon = line.find(':');
        if (colon != std::string::npos)
            headers[toLower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }

    if (!isLocalHost(headers["host"]) || !allowedOrigin(headers["origin"])) {
        sendResponse(client, 403, "Forbidden");
        return;
    }
    const std::string path = target.substr(0, target.find('?'));
    if (path != "/mcp" && path != "/") {
        sendResponse(client, 404, "Not Found");
        return;
    }
    // No server-to-client stream (GET) and no sessions to end (DELETE).
    if (method != "POST") {
        sendResponse(client, 405, "Method Not Allowed", {}, "Allow: POST\r\n");
        return;
    }
    if (headers.count("transfer-encoding")) {
        sendResponse(client, 411, "Length Required");
        return;
    }
    size_t contentLength = 0;
    try {
        contentLength = std::stoull(headers["content-length"]);
    }
    catch (const std::exception&) {
        sendResponse(client, 411, "Length Required");
        return;
    }
    if (contentLength > kMaxBodySize) {
        sendResponse(client, 413, "Payload Too Large");
        return;
    }
    if (toLower(headers["expect"]) == "100-continue")
        sendAll(client, "HTTP/1.1 100 Continue\r\n\r\n");
    std::string body = data.substr(headerEnd + 4);
    while (body.size() < contentLength) {
        const int n = recv(client, buffer, sizeof(buffer), 0);
        if (n <= 0)
            return;
        body.append(buffer, static_cast<size_t>(n));
    }
    body.resize(contentLength);

    const json message = json::parse(body, nullptr, false);
    if (message.is_discarded()) {
        sendResponse(client, 400, "Bad Request", rpcError(nullptr, -32700, "Parse error").dump());
        return;
    }
    json response;
    if (message.is_array()) {
        response = json::array();
        for (const json& item : message) {
            json r = handleMessage(item);
            if (!r.is_null())
                response.push_back(std::move(r));
        }
        if (response.empty())
            response = nullptr;
    }
    else {
        response = handleMessage(message);
    }
    if (response.is_null())
        sendResponse(client, 202, "Accepted");
    else
        sendResponse(client, 200, "OK", response.dump(-1, ' ', false, json::error_handler_t::replace));
}

json McpServer::handleMessage(const json& message)
{
    if (!message.is_object())
        return rpcError(nullptr, -32600, "Invalid Request");
    // Responses (we send no requests) and notifications (initialized, cancelled) need no answer.
    if (!message.contains("method") || !message.contains("id"))
        return nullptr;
    const json& id = message["id"];
    try {
        const std::string method = message["method"].get<std::string>();
        const json params = message.value("params", json::object());
        if (method == "initialize") {
            std::string version = kProtocolVersions[0];
            const std::string requested = params.value("protocolVersion", "");
            for (const char* supported : kProtocolVersions) {
                if (requested == supported)
                    version = supported;
            }
            return rpcResult(id, {
                { "protocolVersion", version },
                { "capabilities", { { "tools", { { "listChanged", false } } } } },
                { "serverInfo", { { "name", kServerName }, { "version", "1.0.0" } } },
                { "instructions", m_instructions },
            });
        }
        if (method == "ping")
            return rpcResult(id, json::object());
        if (method == "tools/list")
            return rpcResult(id, { { "tools", m_tools } });
        if (method == "tools/call") {
            const std::string name = params.at("name").get<std::string>();
            const bool known = std::any_of(m_tools.begin(), m_tools.end(),
                [&](const json& tool) { return tool.value("name", "") == name; });
            if (!known)
                return rpcError(id, -32602, "Unknown tool: " + name);
            const json arguments = params.value("arguments", json::object());
            const ToolResult result = callTool(name, arguments.is_null() ? json::object() : arguments);
            return rpcResult(id, {
                { "content", json::array({ { { "type", "text" }, { "text", result.text } } }) },
                { "isError", result.isError },
            });
        }
        return rpcError(id, -32601, "Method not found: " + method);
    }
    catch (const json::exception& e) {
        return rpcError(id, -32602, std::string("Invalid params: ") + e.what());
    }
}
