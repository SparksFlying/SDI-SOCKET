#pragma once
#include <string>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/attributes/scoped_attribute.hpp>
#include <boost/log/trivial.hpp>
#include <cstddef>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <codecvt>
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
 
namespace logging = boost::log;
namespace src = boost::log::sources;
namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;
 
using std::string;
 
BOOST_LOG_ATTRIBUTE_KEYWORD(tag_attr, "Tag", std::string)
BOOST_LOG_ATTRIBUTE_KEYWORD(scope, "Scope", attrs::named_scope::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(timeline, "Timeline", attrs::timer::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(thread_id, "ThreadID", attrs::current_thread_id::value_type)
class Log
{
public:
	Log(const std::string &sLogFilePfx = "Log", const unsigned int nRotSize = 5 * 1024 * 1024);
	~Log();
 
	void fatalLog(const std::wstring & wsTxt);
	void errorLog(const std::wstring & wsTxt);
	void warningLog(const std::wstring & wsTxt);
	void infoLog(const std::wstring & wsTxt);
	void debugLog(const std::wstring & wsTxt);
	void traceLog(const std::wstring & wsTxt);

	inline std::wstring toWideString(const std::string& input)
	{
		return converter.from_bytes(input);
	}

	void infoLog(const std::string& txt){
		infoLog(toWideString(txt));
	}

	void debugLog(const std::string& txt){
		debugLog(toWideString(txt));
	}
 
	#define M_LOG_USE_TIME_LINE BOOST_LOG_SCOPED_THREAD_ATTR("Timeline", boost::log::attributes::timer());
	#define M_LOG_USE_NAMED_SCOPE(named_scope) BOOST_LOG_NAMED_SCOPE(named_scope);
 
private:
	void internLog(const std::wstring & wsTxt, const boost::log::trivial::severity_level eSevLev);
	boost::log::sources::wseverity_logger_mt<boost::log::trivial::severity_level> m_oWsLogger;
	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
};

 

 
 
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
			keywords::file_name = sLogFilePfx + "_%N.log", 
			keywords::rotation_size = nRotSizeInByte,
			keywords::open_mode = std::ios_base::app
		);
 
	boost::shared_ptr< text_sink > sink(new text_sink(backend));
 
	sink->set_formatter
	(
		expr::stream
		<< expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S") << " | "
		<< logging::trivial::severity << "\t| "
		<< thread_id << " | "
		<< "(" << scope << ") "
		<< expr::if_(expr::has_attr(tag_attr))
		[
			expr::stream << "[" << tag_attr << "] "
		]
	<< expr::if_(expr::has_attr(timeline))
		[
			expr::stream << "[" << timeline << "] "
		]
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
 
void Log::fatalLog(const std::wstring & wsTxt)
{
	if (wsTxt.empty())
	{
		return;
	}
 
	internLog(wsTxt, logging::trivial::fatal);
}
 
void Log::errorLog(const std::wstring & wsTxt)
{
	if (wsTxt.empty())
	{
		return;
	}
 
	internLog(wsTxt, logging::trivial::error);
}
 
void Log::warningLog(const std::wstring & wsTxt)
{
	if (wsTxt.empty())
	{
		return;
	}
 
	internLog(wsTxt, logging::trivial::warning);
}
 
void Log::infoLog(const std::wstring & wsTxt)
{
	if (wsTxt.empty())
	{
		return;
	}
 
	internLog(wsTxt, logging::trivial::info);
}
 
void Log::debugLog(const std::wstring & wsTxt)
{
	if (wsTxt.empty())
	{
		return;
	}
 
	internLog(wsTxt, logging::trivial::debug);
}
 
void Log::traceLog(const std::wstring & wsTxt)
{
	if (wsTxt.empty())
	{
		return;
	}
 
	internLog(wsTxt, logging::trivial::trace);
}
 
void Log::internLog(const std::wstring & wsTxt, const logging::trivial::severity_level eSevLev)
{
	if (wsTxt.empty())
	{
		return;
	}
 
	BOOST_LOG_SEV(m_oWsLogger, eSevLev) << wsTxt;
}