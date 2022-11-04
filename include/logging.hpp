#pragma once
#include <string>
#include <map>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/attributes/scoped_attribute.hpp>
#include <boost/log/trivial.hpp>
#include <cstddef>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <codecvt>
#include <mutex>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <boost/locale/generator.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/basic_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup/file.hpp>
#include "config.hpp"
 
namespace logging = boost::log;
namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;

using std::string;
using std::map;
BOOST_LOG_ATTRIBUTE_KEYWORD(tag_attr, "Tag", std::string)
BOOST_LOG_ATTRIBUTE_KEYWORD(scope, "Scope", attrs::named_scope::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(timeline, "Timeline", attrs::timer::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(thread_id, "ThreadID", attrs::current_thread_id::value_type)

#define debug(format, ...) \
    Log::instance().log(logging::trivial::severity_level::debug, __FUNCTION__, __LINE__, format, ##__VA_ARGS__)

#define info(format, ...) \
    Log::instance().log(logging::trivial::severity_level::info, __FUNCTION__, __LINE__, format, ##__VA_ARGS__)

#define warn(format, ...) \
    Log::instance().log(logging::trivial::severity_level::warning, __FUNCTION__, __LINE__, format, ##__VA_ARGS__)

#define error(format, ...) \
    Log::instance().log(logging::trivial::severity_level::error, __FUNCTION__, __LINE__, format, ##__VA_ARGS__)

#define fatal(format, ...) \
    Log::instance().log(logging::trivial::severity_level::fatal, __FUNCTION__, __LINE__, format, ##__VA_ARGS__)


static std::map<logging::trivial::severity_level, string> level2str{
	{logging::trivial::severity_level::debug, "DEBUG"},
	{logging::trivial::severity_level::info, "INFO"},
	{logging::trivial::severity_level::warning, "WARNING"},
	{logging::trivial::severity_level::error, "ERROR"},
	{logging::trivial::severity_level::fatal, "FATAL"}

};

class Log
{
public:
	
	static Log& instance(const string& prefix = "Log")
	{
		static Log olog(prefix);
		return olog;
		// if(m_instance == nullptr){
		// 	m_mutex.lock();
		// 	if(m_instance == nullptr)
		// 	{
		// 		m_instance = new Log(prefix);
		// 	}
		// 	m_mutex.unlock();
		// }
		// return *m_instance;
	}
	
	void setLogFilePrefix(const string& prefix = "Log")
	{

	}

	void setLevel(logging::trivial::severity_level level){
		this->m_level = level;
	}

	void log(logging::trivial::severity_level level, const char* file, int line, const char* format, ...)
	{
		if (m_level > level)
		{
			return;
		}
		string content = "";

		int len = 0;
		len = snprintf(NULL, 0, "%s | %s | %d | ", level2str[level].c_str(), file, line);
		if(len > 0)
		{
			char * buffer = new char[len + 1];
			snprintf(buffer, len + 1, "%s | %s | %d | ", level2str[level].c_str(), file, line);
			buffer[len] = 0;
			content += buffer;
			delete buffer;
		}
		
		va_list arg_ptr;
		va_start(arg_ptr, format);
		len = vsnprintf(NULL, 0, format, arg_ptr);
		va_end(arg_ptr);

		if (len > 0)
		{
			char * buffer = new char[len + 1];
			va_start(arg_ptr, format);
			vsnprintf(buffer, len + 1, format, arg_ptr);
			va_end(arg_ptr);
			buffer[len] = 0;
			content += buffer;
			delete buffer;
		}
		internLog(toWideString(content), level);

		// if(level == logging::trivial::severity_level::debug)
		// {
		// 	std::cout << content << std::endl;
		// }
	}

	inline std::wstring toWideString(const std::string& input)
	{
		return converter.from_bytes(input);
	}

 
	#define M_LOG_USE_TIME_LINE BOOST_LOG_SCOPED_THREAD_ATTR("Timeline", boost::log::attributes::timer());
	#define M_LOG_USE_NAMED_SCOPE(named_scope) BOOST_LOG_NAMED_SCOPE(named_scope);
 
private:
	Log(const std::string &logFilePrefix = "Log", const unsigned int nRotSize = 5 * 1024 * 1024);
	~Log();
	void internLog(const std::wstring & wsTxt, const boost::log::trivial::severity_level eSevLev);
	boost::log::sources::wseverity_logger_mt<boost::log::trivial::severity_level> m_oWsLogger;
	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	logging::trivial::severity_level m_level = logging::trivial::severity_level::debug;
	static Log* m_instance;
	static std::mutex m_mutex;
};

Log* Log::m_instance = nullptr;
std::mutex Log::m_mutex;
 
Log::Log(const string &sLogFilePfx, const unsigned int nRotSizeInByte)
{
	if (sLogFilePfx.empty())
	{
		throw new std::invalid_argument("日志文件前缀为空");
	}
	if (0 == nRotSizeInByte)
	{
		throw new std::invalid_argument("日志文件旋转大小为0");
	}
 
	typedef sinks::synchronous_sink< sinks::text_file_backend > text_sink;
	boost::shared_ptr< sinks::text_file_backend > backend = boost::make_shared< sinks::text_file_backend >(
			keywords::file_name = config::logFolder + "/"+  sLogFilePfx + "_%N.log", 
			keywords::rotation_size = nRotSizeInByte,
			keywords::open_mode = std::ios_base::app
		);
	boost::shared_ptr< text_sink > sink(new text_sink(backend));
 
	sink->set_formatter
	(
		expr::stream
		<< expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S") << " | "
		<< expr::message
		);
 
	// The sink will perform character code conversion as needed, according to the locale set with imbue()
	std::locale loc = boost::locale::generator()("en_US.UTF-8");
	sink->imbue(loc);
 
	logging::core::get()->add_sink(sink);
 
	// Add attributes
	logging::add_common_attributes();
	logging::core::get()->add_global_attribute("Scope", attrs::named_scope());
}
 
Log::~Log()
{

}

void Log::internLog(const std::wstring & wsTxt, const logging::trivial::severity_level eSevLev)
{
	if (wsTxt.empty())
	{
		return;
	}
 
	BOOST_LOG_SEV(m_oWsLogger, eSevLev) << wsTxt;
}